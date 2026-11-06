use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_types::chat_message::ChatMessage;
use llama_cpp_bindings_types::chat_message_content::ChatMessageContent;
use llama_cpp_bindings_types::chat_tool::ChatTool;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;
use serde_json::Value;

use crate::chat_assistant_prefill::ChatAssistantPrefill;
use crate::chat_assistant_turn::ChatAssistantTurn;
use crate::chat_grammar_trigger::ChatGrammarTrigger;
use crate::chat_output::ChatOutput;
use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_prompt_part::ChatPromptPart;
use crate::chat_reading::ChatReading;
use crate::chat_session_options::ChatSessionOptions;
use crate::chat_templates::ChatTemplates;
use crate::cpp_text_view::cpp_text_view;
use crate::error::chat_session_error::ChatSessionError;
use crate::error::parse_chat_message_error::ParseChatMessageError;
use crate::lazy_grammar_triggers::LazyGrammarTriggers;
use crate::model::LlamaModel;
use crate::model::add_bos::AddBos;
use crate::model::parse_special_tokens::ParseSpecialTokens;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

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

fn session_text(text: *const c_char, length: usize) -> Result<String, ChatSessionError> {
    unsafe { cpp_text_view(text, length) }
        .map(str::to_owned)
        .map_err(ChatSessionError::SessionTextNotUtf8)
}

const fn contract_violation(operation: &'static str, detail: &'static str) -> FfiContractError {
    FfiContractError { operation, detail }
}

fn chat_session_create_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_session_create_status,
    out_session: llama_cpp_bindings_sys::llama_rs_chat_session_handle,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_rs_chat_session>, ChatSessionError> {
    const OPERATION: &str = "llama_rs_chat_session_create";
    let reported_message = || {
        unsafe {
            read_and_free_cpp_string(
                out_error,
                OPERATION,
                "reported a failure without an error message",
            )
        }
        .map_err(ChatSessionError::from)
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_OK => NonNull::new(out_session)
            .ok_or_else(|| {
                contract_violation(OPERATION, "success status contained a null session").into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST => {
            Err(ChatSessionError::TemplateCannotServeRequest {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(ChatSessionError::Reported {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ChatSessionError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ChatSessionError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_TEMPLATES_ARG => {
            Err(contract_violation(OPERATION, "was given a null templates argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_VOCAB_ARG => {
            Err(contract_violation(OPERATION, "was given a null vocab argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_MESSAGES_JSON_ARG => {
            Err(contract_violation(OPERATION, "was given a null messages_json argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_TOOLS_JSON_ARG => {
            Err(contract_violation(OPERATION, "was given a null tools_json argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_SESSION_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_session argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_ERROR_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_error argument").into())
        }
        other => Err(FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

fn chat_session_read_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_session_read_status,
    out_reading: llama_cpp_bindings_sys::llama_rs_chat_reading_handle,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>, ParseChatMessageError> {
    const OPERATION: &str = "llama_rs_chat_session_read";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_OK => NonNull::new(out_reading)
            .ok_or_else(|| {
                contract_violation(OPERATION, "success status contained a null reading").into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(ParseChatMessageError::Reported {
                message: unsafe {
                    read_and_free_cpp_string(
                        out_error,
                        OPERATION,
                        "reported a thrown C++ exception without an error message",
                    )
                }?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_SESSION_ARG => {
            Err(contract_violation(OPERATION, "was given a null session argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_TEXT_ARG => {
            Err(contract_violation(OPERATION, "was given a null text argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_BYTE_TOKENS_ARG => {
            Err(contract_violation(OPERATION, "was given a null byte_tokens argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_OUT_READING_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_reading argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_OUT_ERROR_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_error argument").into())
        }
        other => Err(FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

/// One chat request rendered by the model's chat template the way llama-server renders it.
///
/// The session holds the prompt, the grammar that constrains tool calls, and the parser of the
/// model's answer. It owns everything llama.cpp prepared for the request, so it can be moved to
/// the thread that generates the answer.
pub struct ChatSession {
    session: NonNull<llama_cpp_bindings_sys::llama_rs_chat_session>,
}

unsafe impl Send for ChatSession {}

impl ChatSession {
    /// Renders `messages` and `tools` through `templates`: the model's own
    /// [`LlamaModel::chat_templates`], or [`ChatTemplates::with_override`].
    ///
    /// # Errors
    ///
    /// Returns the [`ChatSessionError`] describing why the chat template cannot serve the request.
    pub fn new(
        model: &LlamaModel,
        templates: &ChatTemplates,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        options: &ChatSessionOptions,
    ) -> Result<Self, ChatSessionError> {
        let messages_json = messages_json(messages, &options.assistant_turn);
        let tools_json = tools_json(tools);
        let mut out_session: llama_cpp_bindings_sys::llama_rs_chat_session_handle = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_create(
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

        chat_session_create_status_to_result(status, out_session, out_error)
            .map(|session| Self { session })
    }

    fn text_of(
        &self,
        read: impl FnOnce(
            *mut llama_cpp_bindings_sys::llama_rs_chat_session,
            *mut usize,
        ) -> *const c_char,
    ) -> Result<String, ChatSessionError> {
        let mut length = 0;
        let text = read(self.session.as_ptr(), &raw mut length);

        session_text(text, length)
    }

    fn texts_of(
        &self,
        count: usize,
        read: impl Fn(
            *mut llama_cpp_bindings_sys::llama_rs_chat_session,
            usize,
            *mut usize,
        ) -> *const c_char,
    ) -> Result<Vec<String>, ChatSessionError> {
        (0..count)
            .map(|index| self.text_of(|session, out_length| read(session, index, out_length)))
            .collect()
    }

    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when the rendered prompt is not UTF-8.
    pub fn prompt_parts(&self) -> Result<Vec<ChatPromptPart>, ChatSessionError> {
        let part_count = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_prompt_part_count(self.session.as_ptr())
        };

        (0..part_count)
            .map(|index| {
                let text = self.text_of(|session, out_length| unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_session_prompt_part_text(
                        session, index, out_length,
                    )
                })?;
                let is_input = unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_session_prompt_part_is_input(
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

    /// The rendered prompt as one string, for [`crate::mtmd::mtmd_context::MtmdContext::tokenize`], which
    /// finds the media markers in it the way llama-server hands a multimodal prompt to mtmd.
    ///
    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when the rendered prompt is not UTF-8.
    pub fn prompt(&self) -> Result<String, ChatSessionError> {
        Ok(self
            .prompt_parts()?
            .iter()
            .map(ChatPromptPart::text)
            .collect())
    }

    /// Tokenizes the rendered prompt part by part with the vocabulary of `model`, the model the
    /// session was created for: control tokens are recognized in the template's own text but
    /// never in user input, and BOS is added once before the first part when the vocabulary asks
    /// for it.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatSessionError`] of reading or tokenizing a prompt part.
    pub fn tokenize_prompt(&self, model: &LlamaModel) -> Result<Vec<LlamaToken>, ChatSessionError> {
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

            tokens.extend(model.str_to_token(text, add_bos, parse_special)?);
        }

        Ok(tokens)
    }

    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when the generation prompt is not UTF-8.
    pub fn generation_prompt(&self) -> Result<String, ChatSessionError> {
        self.text_of(|session, out_length| unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_generation_prompt(session, out_length)
        })
    }

    #[must_use]
    pub fn generation_prompt_tokens(&self) -> Vec<LlamaToken> {
        let count = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_generation_prompt_token_count(
                self.session.as_ptr(),
            )
        };

        (0..count)
            .map(|index| {
                LlamaToken::new(unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_session_generation_prompt_token(
                        self.session.as_ptr(),
                        index,
                    )
                })
            })
            .collect()
    }

    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when the grammar is not UTF-8.
    pub fn grammar(&self) -> Result<String, ChatSessionError> {
        self.text_of(|session, out_length| unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_grammar(session, out_length)
        })
    }

    #[must_use]
    pub fn grammar_is_lazy(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_session_grammar_lazy(self.session.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when a trigger is not UTF-8, or
    /// [`ChatSessionError::FfiStatus`] for a trigger type unknown to this binding.
    pub fn grammar_triggers(&self) -> Result<Vec<ChatGrammarTrigger>, ChatSessionError> {
        let count = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_grammar_trigger_count(
                self.session.as_ptr(),
            )
        };

        (0..count)
            .map(|index| {
                let trigger_type = unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_session_grammar_trigger_type(
                        self.session.as_ptr(),
                        index,
                    )
                };
                let value = self.text_of(|session, out_length| unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_session_grammar_trigger_value(
                        session, index, out_length,
                    )
                })?;

                Ok(ChatGrammarTrigger::from_wrapper(
                    trigger_type,
                    value,
                    || {
                        LlamaToken::new(unsafe {
                            llama_cpp_bindings_sys::llama_rs_chat_session_grammar_trigger_token(
                                self.session.as_ptr(),
                                index,
                            )
                        })
                    },
                )?)
            })
            .collect()
    }

    #[must_use]
    pub fn preserved_tokens(&self) -> Vec<LlamaToken> {
        let count = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_preserved_token_count(
                self.session.as_ptr(),
            )
        };

        (0..count)
            .map(|index| {
                LlamaToken::new(unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_session_preserved_token(
                        self.session.as_ptr(),
                        index,
                    )
                })
            })
            .collect()
    }

    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when the tag is not UTF-8.
    pub fn thinking_start_tag(&self) -> Result<String, ChatSessionError> {
        self.text_of(|session, out_length| unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_thinking_start_tag(session, out_length)
        })
    }

    /// # Errors
    ///
    /// Returns [`ChatSessionError::SessionTextNotUtf8`] when a tag is not UTF-8.
    pub fn thinking_end_tags(&self) -> Result<Vec<String>, ChatSessionError> {
        self.texts_of(
            unsafe {
                llama_cpp_bindings_sys::llama_rs_chat_session_thinking_end_tag_count(
                    self.session.as_ptr(),
                )
            },
            |session, index, out_length| unsafe {
                llama_cpp_bindings_sys::llama_rs_chat_session_thinking_end_tag(
                    session, index, out_length,
                )
            },
        )
    }

    /// The sampler that constrains generation to the request's tool calls with the vocabulary of
    /// `model`, the model the session was created for, or `None` when the request needs no
    /// grammar. A grammar that is not lazy starts after the generation prompt, so it accepts the
    /// generation-prompt tokens first.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatSessionError`] of reading the grammar, escaping a trigger word, building
    /// the sampler, or accepting a generation-prompt token.
    pub fn grammar_sampler(
        &self,
        model: &LlamaModel,
    ) -> Result<Option<LlamaSampler>, ChatSessionError> {
        let grammar = self.grammar()?;

        if grammar.is_empty() {
            return Ok(None);
        }

        if !self.grammar_is_lazy() {
            let mut sampler = LlamaSampler::grammar(model, &grammar, GRAMMAR_ROOT)?;

            for token in self.generation_prompt_tokens() {
                sampler.accept(token)?;
            }

            return Ok(Some(sampler));
        }

        let LazyGrammarTriggers { patterns, tokens } =
            LazyGrammarTriggers::of(self.grammar_triggers()?)?;

        Ok(Some(LlamaSampler::grammar_lazy(
            model,
            &grammar,
            GRAMMAR_ROOT,
            &patterns,
            &tokens,
        )?))
    }

    /// Reads the model's answer with the template's parser: the reading attributes the answer's
    /// bytes to reasoning, content, and tool calls, and builds the parsed message on demand.
    ///
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when llama.cpp fails while parsing; an answer that does
    /// not match the template's format is a reading whose classification says so.
    pub fn read(
        &self,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<ChatReading, ParseChatMessageError> {
        let mut out_reading: llama_cpp_bindings_sys::llama_rs_chat_reading_handle = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_read(
                self.session.as_ptr(),
                output.text().as_ptr().cast::<c_char>(),
                output.text().len(),
                output.byte_tokens().as_ptr(),
                completeness == ChatOutputCompleteness::Partial,
                &raw mut out_reading,
                &raw mut out_error,
            )
        };

        chat_session_read_status_to_result(status, out_reading, out_error)
            .and_then(|reading| ChatReading::take(reading, output, completeness))
    }
}

impl Drop for ChatSession {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_session_free(self.session.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_types::chat_message::ChatMessage;
    use llama_cpp_bindings_types::chat_message_content::ChatMessageContent;
    use llama_cpp_bindings_types::chat_tool::ChatTool;
    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
    use serde_json::json;

    use super::chat_session_create_status_to_result;
    use super::chat_session_read_status_to_result;
    use super::messages_json;
    use super::tools_json;
    use crate::chat_assistant_prefill::ChatAssistantPrefill;
    use crate::chat_assistant_turn::ChatAssistantTurn;
    use crate::error::chat_session_error::ChatSessionError;
    use crate::error::parse_chat_message_error::ParseChatMessageError;

    type CreateRefusal = (
        llama_cpp_bindings_sys::llama_rs_chat_session_create_status,
        fn(String) -> ChatSessionError,
    );

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

    #[test]
    fn creating_a_session_reports_each_null_argument_and_a_null_created_session_as_a_contract_violation()
     {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_TEMPLATES_ARG,
                "was given a null templates argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_VOCAB_ARG,
                "was given a null vocab argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_MESSAGES_JSON_ARG,
                "was given a null messages_json argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_TOOLS_JSON_ARG,
                "was given a null tools_json argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_SESSION_ARG,
                "was given a null out_session argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_OK,
                "success status contained a null session",
            ),
        ] {
            assert_eq!(
                chat_session_create_status_to_result(status, ptr::null_mut(), ptr::null_mut())
                    .err(),
                Some(ChatSessionError::FfiContract(FfiContractError {
                    operation: "llama_rs_chat_session_create",
                    detail
                }))
            );
        }
    }

    #[test]
    fn creating_a_session_reports_why_llama_cpp_refused_the_request() {
        let refusals: [CreateRefusal; 2] = [
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
                |message| ChatSessionError::TemplateCannotServeRequest { message },
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                |message| ChatSessionError::Reported { message },
            ),
        ];

        for (status, expected_error) in refusals {
            assert_eq!(
                chat_session_create_status_to_result(status, ptr::null_mut(), unsafe {
                    llama_cpp_bindings_sys::llama_rs_string_dup(c"the reason".as_ptr())
                })
                .err(),
                Some(expected_error("the reason".to_owned()))
            );
        }
    }

    #[test]
    fn creating_a_session_reports_a_refusal_without_a_message_as_a_contract_violation() {
        let refusals = [
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
        ];

        assert_eq!(
            refusals.map(|status| {
                chat_session_create_status_to_result(status, ptr::null_mut(), ptr::null_mut()).err()
            }),
            refusals.map(|_status| {
                Some(ChatSessionError::CppString(CppStringError::Missing(
                    FfiContractError {
                        operation: "llama_rs_chat_session_create",
                        detail: "reported a failure without an error message",
                    },
                )))
            })
        );
    }

    #[test]
    fn creating_a_session_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| chat_session_create_status_to_result(
                status,
                ptr::null_mut(),
                ptr::null_mut()
            )
            .err()),
            [
                Some(ChatSessionError::NotEnoughMemory),
                Some(ChatSessionError::LlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn creating_a_session_preserves_an_unknown_status() {
        assert_eq!(
            chat_session_create_status_to_result(255, ptr::null_mut(), ptr::null_mut()).err(),
            Some(ChatSessionError::FfiStatus(FfiStatusError {
                operation: "llama_rs_chat_session_create",
                code: 255
            }))
        );
    }

    #[test]
    fn reading_an_answer_maps_every_failure_status() {
        let read = |status, out_error| {
            chat_session_read_status_to_result(status, ptr::null_mut(), out_error).err()
        };
        let contract = |detail| {
            Some(ParseChatMessageError::FfiContract(FfiContractError {
                operation: "llama_rs_chat_session_read",
                detail,
            }))
        };

        assert_eq!(
            [
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_OK,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"the parser threw".as_ptr()) }
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_SESSION_ARG,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_TEXT_ARG,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_BYTE_TOKENS_ARG,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_OUT_READING_ARG,
                    ptr::null_mut()
                ),
                read(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_READ_NULL_OUT_ERROR_ARG,
                    ptr::null_mut()
                ),
                read(255, ptr::null_mut()),
            ],
            [
                contract("success status contained a null reading"),
                Some(ParseChatMessageError::Reported {
                    message: "the parser threw".to_owned()
                }),
                Some(ParseChatMessageError::CppString(CppStringError::Missing(FfiContractError {
                    operation: "llama_rs_chat_session_read",
                    detail: "reported a thrown C++ exception without an error message",
                }))),
                Some(ParseChatMessageError::NotEnoughMemory),
                Some(ParseChatMessageError::LlamaCppOutOfMemory),
                contract("was given a null session argument"),
                contract("was given a null text argument"),
                contract("was given a null byte_tokens argument"),
                contract("was given a null out_reading argument"),
                contract("was given a null out_error argument"),
                Some(ParseChatMessageError::FfiStatus(FfiStatusError { operation: "llama_rs_chat_session_read", code: 255 })),
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
}
