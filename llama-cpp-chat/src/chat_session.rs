use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use serde_json::Value;

use llama_cpp_bindings_sys::bindings::llama_rs_chat_grammar_trigger_type;
use llama_cpp_bindings_sys::bindings::llama_rs_chat_session_create_status;
use llama_cpp_bindings_sys::bindings::llama_rs_chat_session_free;
use llama_cpp_bindings_sys::bindings::llama_rs_chat_session_read_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;
use llama_cpp_bindings_sys::cpp_text_view::cpp_text_view;
use llama_cpp_bindings_types::chat_message::ChatMessage;
use llama_cpp_bindings_types::chat_message_content::ChatMessageContent;
use llama_cpp_bindings_types::chat_tool::ChatTool;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::chat_templates::ChatTemplates;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_sampler::chat_grammar_trigger::ChatGrammarTrigger;
use llama_cpp_sampler::lazy_grammar_triggers::LazyGrammarTriggers;
use llama_cpp_sampler::llama_sampler::LlamaSampler;

use crate::chat_assistant_prefill::ChatAssistantPrefill;
use crate::chat_assistant_turn::ChatAssistantTurn;
use crate::chat_error::ChatError;
use crate::chat_output::ChatOutput;
use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_prompt_part::ChatPromptPart;
use crate::chat_reading::ChatReading;
use crate::chat_session_options::ChatSessionOptions;

const ASSISTANT_ROLE: &str = "assistant";
const GRAMMAR_ROOT: &str = "root";

fn continued_assistant_message(assistant_turn: &ChatAssistantTurn) -> Option<ChatMessage> {
    match assistant_turn {
        ChatAssistantTurn::Omitted | ChatAssistantTurn::New => None,
        ChatAssistantTurn::ContinueReasoning(reasoning_content) => Some(ChatMessage::Message {
            role: ASSISTANT_ROLE.to_owned(),
            content: ChatMessageContent::Text(String::new()),
            reasoning_content: Some(reasoning_content.clone()),
        }),
        ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
            reasoning_content,
            content,
        }) => Some(ChatMessage::Message {
            role: ASSISTANT_ROLE.to_owned(),
            content: ChatMessageContent::Text(content.clone()),
            reasoning_content: Some(reasoning_content.clone()),
        }),
    }
}

fn messages_json(messages: &[ChatMessage], assistant_turn: &ChatAssistantTurn) -> String {
    messages
        .iter()
        .chain(continued_assistant_message(assistant_turn).as_ref())
        .map(Value::from)
        .collect::<Value>()
        .to_string()
}

fn tools_json(tools: &[ChatTool]) -> String {
    tools.iter().map(Value::from).collect::<Value>().to_string()
}

fn session_text(text: *const c_char, length: usize) -> Result<String, ChatError> {
    unsafe { cpp_text_view(text, length) }
        .map(str::to_owned)
        .map_err(ChatError::SessionTextNotUtf8)
}

fn grammar_trigger_of(
    trigger_type: llama_rs_chat_grammar_trigger_type,
    value: String,
    token: impl FnOnce() -> LlamaToken,
) -> Result<ChatGrammarTrigger, ChatError> {
    match trigger_type {
        llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_TOKEN => {
            Ok(ChatGrammarTrigger::Token(token()))
        }
        llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_WORD => {
            Ok(ChatGrammarTrigger::Word(value))
        }
        llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN => {
            Ok(ChatGrammarTrigger::Pattern(value))
        }
        llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL => {
            Ok(ChatGrammarTrigger::PatternFull(value))
        }
        llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED => {
            Err(ChatError::GrammarTriggerTypeUnrecognized)
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_CHAT_SESSION_CREATE_OK` `out_session` must be the non-null handle the C++ side
/// created, and on `LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST` or
/// `LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null
/// message it allocated; the message is freed here.
unsafe fn chat_session_create_status_to_result(
    status: llama_rs_chat_session_create_status,
    out_session: llama_cpp_bindings_sys::bindings::llama_rs_chat_session_handle,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_session>, ChatError> {
    const OPERATION: &str = "llama_rs_chat_session_create";

    match status {
        llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_session) })
        }
        llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error(OPERATION, |message| {
                    ChatError::TemplateCannotServeRequest { message }
                }))
        }
        llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error(OPERATION, |message| {
                    ChatError::SessionReported { message }
                }))
        }
        llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ChatError::SessionNotEnoughMemory)
        }
        llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ChatError::SessionLlamaCppOutOfMemory)
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_CHAT_SESSION_READ_OK` `out_reading` must be the non-null handle the C++ side
/// created, and on `LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be
/// the non-null message it allocated; the message is freed here.
unsafe fn chat_session_read_status_to_result(
    status: llama_rs_chat_session_read_status,
    out_reading: llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_handle,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>, ChatError> {
    match status {
        llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_reading) })
        }
        llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_chat_session_read", |message| {
                    ChatError::ParseReported { message }
                }))
        }
        llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ChatError::ParseNotEnoughMemory)
        }
        llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ChatError::ParseLlamaCppOutOfMemory)
        }
    }
}

/// One chat request rendered by the model's chat template the way llama-server renders it.
///
/// The session holds the prompt, the grammar that constrains tool calls, and the parser of the
/// model's answer. It owns everything llama.cpp prepared for the request, so it can be moved to
/// the thread that generates the answer.
pub struct ChatSession {
    session: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_session>,
}

unsafe impl Send for ChatSession {}

impl ChatSession {
    /// Renders `messages` and `tools` through `templates`: the model's own
    /// [`LlamaModel::chat_templates`], or [`LlamaModel::chat_templates_with_override`].
    ///
    /// # Errors
    ///
    /// Returns the [`ChatError`] describing why the chat template cannot serve the request.
    pub fn new(
        model: &LlamaModel,
        templates: &ChatTemplates,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        options: &ChatSessionOptions,
    ) -> Result<Self, ChatError> {
        let messages_json = messages_json(messages, &options.assistant_turn);
        let tools_json = tools_json(tools);
        let mut out_session: llama_cpp_bindings_sys::bindings::llama_rs_chat_session_handle =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_create(
                templates.as_ptr(),
                model.vocab_ptr(),
                messages_json.as_ptr().cast::<c_char>(),
                messages_json.len(),
                tools_json.as_ptr().cast::<c_char>(),
                tools_json.len(),
                options.into(),
                &raw mut out_session,
                &raw mut out_error,
            )
        };

        unsafe { chat_session_create_status_to_result(status, out_session, out_error) }
            .map(|session| Self { session })
    }

    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when the generation prompt is not UTF-8.
    pub fn generation_prompt(&self) -> Result<String, ChatError> {
        self.text_of(|session, out_length| unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_generation_prompt(
                session, out_length,
            )
        })
    }

    #[must_use]
    pub fn generation_prompt_tokens(&self) -> Vec<LlamaToken> {
        let count = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_generation_prompt_token_count(
                self.session.as_ptr(),
            )
        };

        (0..count)
            .map(|index| {
                LlamaToken::new(unsafe {
                    llama_cpp_bindings_sys::bindings::llama_rs_chat_session_generation_prompt_token(
                        self.session.as_ptr(),
                        index,
                    )
                })
            })
            .collect()
    }

    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when the grammar is not UTF-8.
    pub fn grammar(&self) -> Result<String, ChatError> {
        self.text_of(|session, out_length| unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_grammar(session, out_length)
        })
    }

    #[must_use]
    pub fn grammar_is_lazy(&self) -> bool {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_grammar_lazy(
                self.session.as_ptr(),
            )
        }
    }

    /// The sampler that constrains generation to the request's tool calls with the vocabulary of
    /// `model`, the model the session was created for, or `None` when the request needs no
    /// grammar. A grammar that is not lazy starts after the generation prompt, so it accepts the
    /// generation-prompt tokens first.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatError`] of reading the grammar, escaping a trigger word, building
    /// the sampler, or accepting a generation-prompt token.
    pub fn grammar_sampler(&self, model: &LlamaModel) -> Result<Option<LlamaSampler>, ChatError> {
        let grammar = self.grammar()?;

        if grammar.is_empty() {
            return Ok(None);
        }

        if !self.grammar_is_lazy() {
            let mut sampler =
                LlamaSampler::grammar(model, &grammar, GRAMMAR_ROOT).map_err(ChatError::Grammar)?;

            for token in self.generation_prompt_tokens() {
                sampler.accept(token).map_err(ChatError::GrammarPrefill)?;
            }

            return Ok(Some(sampler));
        }

        let LazyGrammarTriggers { patterns, tokens } =
            LazyGrammarTriggers::of(self.grammar_triggers()?).map_err(ChatError::Grammar)?;

        LlamaSampler::grammar_lazy(model, &grammar, GRAMMAR_ROOT, &patterns, &tokens)
            .map(Some)
            .map_err(ChatError::Grammar)
    }

    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when a trigger is not UTF-8, or
    /// [`ChatError::GrammarTriggerTypeUnrecognized`] for a trigger type unknown to this binding.
    pub fn grammar_triggers(&self) -> Result<Vec<ChatGrammarTrigger>, ChatError> {
        let count = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_grammar_trigger_count(
                self.session.as_ptr(),
            )
        };

        (0..count)
            .map(|index| {
                let trigger_type = unsafe {
                    llama_cpp_bindings_sys::bindings::llama_rs_chat_session_grammar_trigger_type(
                        self.session.as_ptr(),
                        index,
                    )
                };
                let value = self.text_of(|session, out_length| unsafe {
                    llama_cpp_bindings_sys::bindings::llama_rs_chat_session_grammar_trigger_value(
                        session, index, out_length,
                    )
                })?;

                grammar_trigger_of(trigger_type, value, || {
                    LlamaToken::new(unsafe {
                        llama_cpp_bindings_sys::bindings::llama_rs_chat_session_grammar_trigger_token(
                            self.session.as_ptr(),
                            index,
                        )
                    })
                })
            })
            .collect()
    }

    #[must_use]
    pub fn preserved_tokens(&self) -> Vec<LlamaToken> {
        let count = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_preserved_token_count(
                self.session.as_ptr(),
            )
        };

        (0..count)
            .map(|index| {
                LlamaToken::new(unsafe {
                    llama_cpp_bindings_sys::bindings::llama_rs_chat_session_preserved_token(
                        self.session.as_ptr(),
                        index,
                    )
                })
            })
            .collect()
    }

    /// The rendered prompt as one string, for `MtmdInputChunks::tokenize`, which
    /// finds the media markers in it the way llama-server hands a multimodal prompt to mtmd.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when the rendered prompt is not UTF-8.
    pub fn prompt(&self) -> Result<String, ChatError> {
        Ok(self
            .prompt_parts()?
            .iter()
            .map(ChatPromptPart::text)
            .collect())
    }

    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when the rendered prompt is not UTF-8.
    pub fn prompt_parts(&self) -> Result<Vec<ChatPromptPart>, ChatError> {
        let part_count = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_prompt_part_count(
                self.session.as_ptr(),
            )
        };

        (0..part_count)
            .map(|index| {
                let text = self.text_of(|session, out_length| unsafe {
                    llama_cpp_bindings_sys::bindings::llama_rs_chat_session_prompt_part_text(
                        session, index, out_length,
                    )
                })?;
                let is_input = unsafe {
                    llama_cpp_bindings_sys::bindings::llama_rs_chat_session_prompt_part_is_input(
                        self.session.as_ptr(),
                        index,
                    )
                };

                Ok(if is_input {
                    ChatPromptPart::Input(text)
                } else {
                    ChatPromptPart::Template(text)
                })
            })
            .collect()
    }

    /// Reads the model's answer with the template's parser: the reading attributes the answer's
    /// bytes to reasoning, content, and tool calls, and builds the parsed message on demand.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when llama.cpp fails while parsing; an answer that does
    /// not match the template's format is a reading whose classification says so.
    pub fn read(
        &self,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<ChatReading, ChatError> {
        let mut out_reading: llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_handle =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_read(
                self.session.as_ptr(),
                output.text().as_ptr().cast::<c_char>(),
                output.text().len(),
                output.byte_tokens().as_ptr(),
                completeness == ChatOutputCompleteness::Partial,
                &raw mut out_reading,
                &raw mut out_error,
            )
        };

        unsafe { chat_session_read_status_to_result(status, out_reading, out_error) }
            .and_then(|reading| unsafe { ChatReading::take(reading, output, completeness) })
    }

    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when a tag is not UTF-8.
    pub fn thinking_end_tags(&self) -> Result<Vec<String>, ChatError> {
        self.texts_of(
            unsafe {
                llama_cpp_bindings_sys::bindings::llama_rs_chat_session_thinking_end_tag_count(
                    self.session.as_ptr(),
                )
            },
            |session, index, out_length| unsafe {
                llama_cpp_bindings_sys::bindings::llama_rs_chat_session_thinking_end_tag(
                    session, index, out_length,
                )
            },
        )
    }

    /// # Errors
    ///
    /// Returns [`ChatError::SessionTextNotUtf8`] when the tag is not UTF-8.
    pub fn thinking_start_tag(&self) -> Result<String, ChatError> {
        self.text_of(|session, out_length| unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_session_thinking_start_tag(
                session, out_length,
            )
        })
    }

    /// Tokenizes the rendered prompt part by part with the vocabulary of `model`, the model the
    /// session was created for: control tokens are recognized in the template's own text but
    /// never in user input, and BOS is added once before the first part when the vocabulary asks
    /// for it.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatError`] of reading or tokenizing a prompt part.
    pub fn tokenize_prompt(&self, model: &LlamaModel) -> Result<Vec<LlamaToken>, ChatError> {
        let mut tokens = Vec::new();

        for (part_index, part) in self.prompt_parts()?.iter().enumerate() {
            let add_bos = if part_index == 0 {
                AddBos::Always
            } else {
                AddBos::Never
            };
            let (text, parse_special) = match part {
                ChatPromptPart::Template(text) => (text, ParseSpecialTokens::Always),
                ChatPromptPart::Input(text) => (text, ParseSpecialTokens::Never),
            };

            tokens.extend(
                model
                    .str_to_token(text, add_bos, parse_special)
                    .map_err(ChatError::PromptTokenization)?,
            );
        }

        Ok(tokens)
    }

    fn text_of(
        &self,
        read: impl FnOnce(
            *mut llama_cpp_bindings_sys::bindings::llama_rs_chat_session,
            *mut usize,
        ) -> *const c_char,
    ) -> Result<String, ChatError> {
        let mut length = 0;
        let text = read(self.session.as_ptr(), &raw mut length);

        session_text(text, length)
    }

    fn texts_of(
        &self,
        count: usize,
        read: impl Fn(
            *mut llama_cpp_bindings_sys::bindings::llama_rs_chat_session,
            usize,
            *mut usize,
        ) -> *const c_char,
    ) -> Result<Vec<String>, ChatError> {
        (0..count)
            .map(|index| self.text_of(|session, out_length| read(session, index, out_length)))
            .collect()
    }
}

impl Drop for ChatSession {
    fn drop(&mut self) {
        unsafe { llama_rs_chat_session_free(self.session.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use serde_json::json;

    use llama_cpp_bindings_sys::bindings::llama_rs_chat_grammar_trigger_type;
    use llama_cpp_bindings_sys::bindings::llama_rs_chat_session_create_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_chat_session_read_status;
    use llama_cpp_bindings_types::chat_message::ChatMessage;
    use llama_cpp_bindings_types::chat_message_content::ChatMessageContent;
    use llama_cpp_bindings_types::chat_tool::ChatTool;
    use llama_cpp_model::llama_token::LlamaToken;
    use llama_cpp_sampler::chat_grammar_trigger::ChatGrammarTrigger;

    use super::chat_session_create_status_to_result;
    use super::chat_session_read_status_to_result;
    use super::grammar_trigger_of;
    use super::messages_json;
    use super::tools_json;
    use crate::chat_assistant_prefill::ChatAssistantPrefill;
    use crate::chat_assistant_turn::ChatAssistantTurn;
    use crate::chat_error::ChatError;

    fn trigger_of(
        trigger_type: llama_rs_chat_grammar_trigger_type,
    ) -> Result<ChatGrammarTrigger, ChatError> {
        grammar_trigger_of(trigger_type, "<tool_call>".to_owned(), || {
            LlamaToken::new(151_657)
        })
    }

    fn parsed_json(json_text: &str) -> serde_json::Value {
        serde_json::from_str(json_text).unwrap()
    }

    fn user_says_hi() -> ChatMessage {
        ChatMessage::Message {
            role: "user".to_owned(),
            content: ChatMessageContent::Text("hi".to_owned()),
            reasoning_content: None,
        }
    }

    struct CreateRefusal {
        status: llama_rs_chat_session_create_status,
        expected_error: fn(String) -> ChatError,
    }

    #[test]
    fn creating_a_session_reports_why_llama_cpp_refused_the_request() {
        let refusals: [CreateRefusal; 2] = [
            CreateRefusal {
                status:
                    llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
                expected_error: |message| ChatError::TemplateCannotServeRequest { message },
            },
            CreateRefusal {
                status:
                    llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                expected_error: |message| ChatError::SessionReported { message },
            },
        ];

        for CreateRefusal {
            status,
            expected_error,
        } in refusals
        {
            assert_eq!(
                unsafe {
                    chat_session_create_status_to_result(
                        status,
                        ptr::null_mut(),
                        llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                            c"the reason".as_ptr(),
                        ),
                    )
                }
                .err(),
                Some(expected_error("the reason".to_owned()))
            );
        }
    }

    #[test]
    fn creating_a_session_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED,
                llama_rs_chat_session_create_status::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                chat_session_create_status_to_result(status, ptr::null_mut(), ptr::null_mut())
            }
            .err()),
            [
                Some(ChatError::SessionNotEnoughMemory),
                Some(ChatError::SessionLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn reading_an_answer_maps_every_failure_status() {
        let read_error = |status, out_error| {
            unsafe { chat_session_read_status_to_result(status, ptr::null_mut(), out_error) }.err()
        };

        assert_eq!(
            [
                read_error(
                    llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"the parser threw".as_ptr()) }
                ),
                read_error(
                    llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                read_error(
                    llama_rs_chat_session_read_status::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
            ],
            [
                Some(ChatError::ParseReported {
                    message: "the parser threw".to_owned()
                }),
                Some(ChatError::ParseNotEnoughMemory),
                Some(ChatError::ParseLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn a_new_assistant_turn_adds_no_message() {
        assert_eq!(
            parsed_json(&messages_json(&[user_says_hi()], &ChatAssistantTurn::New)),
            json!([{"role": "user", "content": "hi"}])
        );
    }

    #[test]
    fn continuing_reasoning_appends_an_assistant_message_with_that_reasoning() {
        assert_eq!(
            parsed_json(&messages_json(
                &[],
                &ChatAssistantTurn::ContinueReasoning("hmm".to_owned())
            )),
            json!([{"role": "assistant", "content": "", "reasoning_content": "hmm"}])
        );
    }

    #[test]
    fn continuing_content_appends_an_assistant_message_with_the_prefill() {
        assert_eq!(
            parsed_json(&messages_json(
                &[],
                &ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
                    reasoning_content: "done".to_owned(),
                    content: "The".to_owned(),
                })
            )),
            json!([{"role": "assistant", "content": "The", "reasoning_content": "done"}])
        );
    }

    #[test]
    fn a_nul_character_in_a_message_is_escaped_in_the_request() {
        let message = ChatMessage::Message {
            role: "user".to_owned(),
            content: ChatMessageContent::Text("a\0b".to_owned()),
            reasoning_content: None,
        };

        assert_eq!(
            messages_json(&[message], &ChatAssistantTurn::Omitted),
            r#"[{"role":"user","content":"a\u0000b"}]"#
        );
    }

    #[test]
    fn tools_are_an_array_of_openai_functions() {
        assert_eq!(
            parsed_json(&tools_json(&[ChatTool {
                name: "get_time".to_owned(),
                description: "Reports the time".to_owned(),
                parameters: None,
            }])),
            json!([{
                "type": "function",
                "function": {"name": "get_time", "description": "Reports the time"},
            }])
        );
    }

    #[test]
    fn every_recognized_trigger_type_maps_to_its_trigger() {
        assert_eq!(
            [
                llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_TOKEN,
                llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_WORD,
                llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN,
                llama_rs_chat_grammar_trigger_type::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
            ]
            .map(trigger_of),
            [
                Ok(ChatGrammarTrigger::Token(LlamaToken::new(151_657))),
                Ok(ChatGrammarTrigger::Word("<tool_call>".to_owned())),
                Ok(ChatGrammarTrigger::Pattern("<tool_call>".to_owned())),
                Ok(ChatGrammarTrigger::PatternFull("<tool_call>".to_owned())),
            ]
        );
    }
}
