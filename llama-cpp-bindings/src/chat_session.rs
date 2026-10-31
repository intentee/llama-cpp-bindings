use std::ffi::CString;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;
use std::slice;

use serde_json::Value;
use serde_json::json;

use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::chat_assistant_prefill::ChatAssistantPrefill;
use crate::chat_assistant_turn::ChatAssistantTurn;
use crate::chat_classification::ChatClassification;
use crate::chat_grammar_trigger::ChatGrammarTrigger;
use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
use crate::chat_output::ChatOutput;
use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_prompt_part::ChatPromptPart;
use crate::chat_session_options::ChatSessionOptions;
use crate::chat_span::ChatSpan;
use crate::chat_span_kind::ChatSpanKind;
use crate::chat_spans::ChatSpans;
use crate::chat_template_tool_calls::known_marker_candidates::known_marker_candidates;
use crate::cpp_text_view::cpp_text_view;
use crate::error::chat_session_error::ChatSessionError;
use crate::error::parse_chat_message_error::ParseChatMessageError;
use crate::lazy_grammar_triggers::LazyGrammarTriggers;
use crate::model::LlamaModel;
use crate::model::add_bos::AddBos;
use crate::model::llama_chat_message::LlamaChatMessage;
use crate::model::parse_special_tokens::ParseSpecialTokens;
use crate::parsed_chat::take_parsed_chat_message;
use crate::raw_chat_message::RawChatMessage;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::tool_call_format::tool_call_format_outcome::ToolCallFormatOutcome;
use crate::tool_call_format::try_parse;

const GRAMMAR_ROOT: &str = "root";

fn chat_message_json(message: &LlamaChatMessage) -> Result<Value, ChatSessionError> {
    Ok(json!({
        "role": message.role.to_str().map_err(ChatSessionError::MessageNotUtf8)?,
        "content": message.content.to_str().map_err(ChatSessionError::MessageNotUtf8)?,
    }))
}

fn continued_assistant_message_json(assistant_turn: &ChatAssistantTurn) -> Option<Value> {
    match assistant_turn {
        ChatAssistantTurn::Omitted | ChatAssistantTurn::New => None,
        ChatAssistantTurn::ContinueReasoning(reasoning_content) => Some(json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning_content,
        })),
        ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
            reasoning_content,
            content,
        }) => Some(json!({
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
        })),
    }
}

fn messages_json(
    messages: &[LlamaChatMessage],
    assistant_turn: &ChatAssistantTurn,
) -> Result<CString, ChatSessionError> {
    let mut messages = messages
        .iter()
        .map(chat_message_json)
        .collect::<Result<Vec<Value>, ChatSessionError>>()?;

    messages.extend(continued_assistant_message_json(assistant_turn));

    CString::new(Value::Array(messages).to_string()).map_err(ChatSessionError::RequestContainsNul)
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
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_NOT_JSON => {
            Err(ChatSessionError::MessagesNotJson {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_INVALID => {
            Err(ChatSessionError::MessagesInvalid {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_NOT_JSON => {
            Err(ChatSessionError::ToolsNotJson {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_INVALID => {
            Err(ChatSessionError::ToolsInvalid {
                message: reported_message()?,
            })
        }
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

fn chat_session_parse_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_session_parse_status,
    out_handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    out_error: *mut c_char,
    output: &ChatOutput,
    completeness: ChatOutputCompleteness,
) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
    const OPERATION: &str = "llama_rs_chat_session_parse";
    let reported_message = || unsafe {
        read_and_free_cpp_string(
            out_error,
            OPERATION,
            "reported a failure without an error message",
        )
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_OK => NonNull::new(out_handle)
            .ok_or_else(|| {
                contract_violation(OPERATION, "success status contained a null handle").into()
            })
            .and_then(take_parsed_chat_message)
            .map(ChatMessageParseOutcome::Recognized),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_OUTPUT_DOES_NOT_MATCH_FORMAT => {
            Ok(ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                text: output.text().to_owned(),
                completeness,
                ffi_error_message: reported_message()?,
            }))
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(ParseChatMessageError::Reported {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_SESSION_ARG => {
            Err(contract_violation(OPERATION, "was given a null session argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_TEXT_ARG => {
            Err(contract_violation(OPERATION, "was given a null text argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_BYTE_TOKENS_ARG => {
            Err(contract_violation(OPERATION, "was given a null byte_tokens argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_HANDLE_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_handle argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_ERROR_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_error argument").into())
        }
        other => Err(FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

fn read_chat_spans(
    spans: NonNull<llama_cpp_bindings_sys::llama_rs_chat_spans>,
    completeness: ChatOutputCompleteness,
) -> Result<ChatSpans, ParseChatMessageError> {
    let view = unsafe { llama_cpp_bindings_sys::llama_rs_chat_spans_view_of(spans.as_ptr()) };
    let wrapper_spans = NonNull::new(view.spans.cast_mut()).map_or(&[][..], |first_span| unsafe {
        slice::from_raw_parts(first_span.as_ptr(), view.spans_count)
    });
    let read = wrapper_spans
        .iter()
        .map(|wrapper_span| {
            ChatSpanKind::try_from(wrapper_span.kind).map(|kind| ChatSpan {
                kind,
                bytes: wrapper_span.start..wrapper_span.end,
            })
        })
        .collect::<Result<Vec<ChatSpan>, FfiStatusError>>()
        .map(|spans_read| ChatSpans {
            parsed_length: view.parsed_length,
            spans: spans_read,
            completeness,
        })
        .map_err(ParseChatMessageError::from);

    unsafe { llama_cpp_bindings_sys::llama_rs_chat_spans_free(spans.as_ptr()) };

    read
}

fn chat_session_classify_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_session_classify_status,
    out_spans: *mut llama_cpp_bindings_sys::llama_rs_chat_spans,
    out_error: *mut c_char,
    completeness: ChatOutputCompleteness,
) -> Result<ChatClassification, ParseChatMessageError> {
    const OPERATION: &str = "llama_rs_chat_session_classify";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_OK => NonNull::new(out_spans)
            .ok_or_else(|| {
                contract_violation(OPERATION, "success status contained null spans").into()
            })
            .and_then(|spans| read_chat_spans(spans, completeness))
            .map(ChatClassification::Spans),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_OUTPUT_DOES_NOT_MATCH_FORMAT => {
            Ok(ChatClassification::OutputDoesNotMatchFormat(completeness))
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_THREW_CXX_EXCEPTION => {
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
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_SESSION_ARG => {
            Err(contract_violation(OPERATION, "was given a null session argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_TEXT_ARG => {
            Err(contract_violation(OPERATION, "was given a null text argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_BYTE_TOKENS_ARG => {
            Err(contract_violation(OPERATION, "was given a null byte_tokens argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_SPANS_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_spans argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_ERROR_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_error argument").into())
        }
        other => Err(FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

/// One chat request rendered by the model's chat template the way llama-server renders it: the
/// prompt, the grammar that constrains tool calls, and the parser of the model's answer.
pub struct ChatSession<'model> {
    model: &'model LlamaModel,
    session: NonNull<llama_cpp_bindings_sys::llama_rs_chat_session>,
    supports_tool_calls: bool,
}

unsafe impl Send for ChatSession<'_> {}

impl<'model> ChatSession<'model> {
    /// Renders `messages` with `tools_json`, an `OpenAI` tools array, through the model's chat
    /// template.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatSessionError`] describing why the model's chat template cannot serve the
    /// request.
    pub fn new(
        model: &'model LlamaModel,
        messages: &[LlamaChatMessage],
        tools_json: &str,
        options: &ChatSessionOptions,
    ) -> Result<Self, ChatSessionError> {
        let templates = model.chat_templates()?;
        let messages_json = messages_json(messages, &options.assistant_turn)?;
        let tools_json = CString::new(tools_json).map_err(ChatSessionError::RequestContainsNul)?;
        let mut out_session: llama_cpp_bindings_sys::llama_rs_chat_session_handle = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_create(
                templates.as_ptr(),
                model.vocab_ptr(),
                messages_json.as_ptr(),
                tools_json.as_ptr(),
                options.into(),
                &raw mut out_session,
                &raw mut out_error,
            )
        };

        chat_session_create_status_to_result(status, out_session, out_error).map(|session| Self {
            model,
            session,
            supports_tool_calls: templates.supports_tool_calls(),
        })
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

    /// Tokenizes the rendered prompt part by part: control tokens are recognized in the
    /// template's own text but never in user input, and BOS is added once before the first part
    /// when the vocabulary asks for it.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatSessionError`] of reading or tokenizing a prompt part.
    pub fn tokenize_prompt(&self) -> Result<Vec<LlamaToken>, ChatSessionError> {
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

            tokens.extend(self.model.str_to_token(text, add_bos, parse_special)?);
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

    /// Whether the template this request rendered with can render tool calls. When it cannot,
    /// [`Self::parse`] recognizes tool calls in the formats other templates use.
    #[must_use]
    pub const fn supports_tool_calls(&self) -> bool {
        self.supports_tool_calls
    }

    #[must_use]
    pub const fn model(&self) -> &'model LlamaModel {
        self.model
    }

    /// The sampler that constrains generation to the request's tool calls, or `None` when the
    /// request needs no grammar. A grammar that is not lazy starts after the generation prompt,
    /// so it accepts the generation-prompt tokens first.
    ///
    /// # Errors
    ///
    /// Returns the [`ChatSessionError`] of reading the grammar, escaping a trigger word, building
    /// the sampler, or accepting a generation-prompt token.
    pub fn grammar_sampler(&self) -> Result<Option<LlamaSampler>, ChatSessionError> {
        let grammar = self.grammar()?;

        if grammar.is_empty() {
            return Ok(None);
        }

        if !self.grammar_is_lazy() {
            let mut sampler = LlamaSampler::grammar(self.model, &grammar, GRAMMAR_ROOT)?;

            for token in self.generation_prompt_tokens() {
                sampler.accept(token)?;
            }

            return Ok(Some(sampler));
        }

        let LazyGrammarTriggers { patterns, tokens } =
            LazyGrammarTriggers::of(self.grammar_triggers()?)?;

        Ok(Some(LlamaSampler::grammar_lazy(
            self.model,
            &grammar,
            GRAMMAR_ROOT,
            &patterns,
            &tokens,
        )?))
    }

    fn parse_with_template(
        &self,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
        let mut out_handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_parse(
                self.session.as_ptr(),
                output.text().as_ptr().cast::<c_char>(),
                output.text().len(),
                output.byte_tokens().as_ptr(),
                completeness == ChatOutputCompleteness::Partial,
                &raw mut out_handle,
                &raw mut out_error,
            )
        };

        chat_session_parse_status_to_result(status, out_handle, out_error, output, completeness)
    }

    fn parse_duck_typed_tool_calls(
        &self,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<Option<ParsedChatMessage>, ParseChatMessageError> {
        for candidate in known_marker_candidates() {
            if let ToolCallFormatOutcome::Parsed(tool_calls) = try_parse(output.text(), &candidate)
            {
                let before_tool_calls = output
                    .text()
                    .find(&candidate.open)
                    .map_or_else(|| output.text(), |open| &output.text()[..open]);

                return Ok(Some(
                    self.parse_with_template(&ChatOutput::from(before_tool_calls), completeness)?
                        .followed_by_tool_calls(tool_calls),
                ));
            }
        }

        Ok(None)
    }

    /// Parses the model's answer with the template's parser. When the template cannot render
    /// tool calls, tool calls in the formats other templates use are recognized first.
    ///
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when llama.cpp fails while parsing; an answer that does
    /// not match the template's format is [`ChatMessageParseOutcome::Unrecognized`].
    pub fn parse(
        &self,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
        if !self.supports_tool_calls
            && let Some(duck_typed) = self.parse_duck_typed_tool_calls(output, completeness)?
        {
            return Ok(ChatMessageParseOutcome::Recognized(duck_typed));
        }

        self.parse_with_template(output, completeness)
    }

    /// Returns how the template's parser attributes the bytes of the model's answer to
    /// reasoning, content, and tool calls.
    ///
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when llama.cpp fails while parsing.
    pub fn classify(
        &self,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<ChatClassification, ParseChatMessageError> {
        let mut out_spans: *mut llama_cpp_bindings_sys::llama_rs_chat_spans = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_session_classify(
                self.session.as_ptr(),
                output.text().as_ptr().cast::<c_char>(),
                output.text().len(),
                output.byte_tokens().as_ptr(),
                completeness == ChatOutputCompleteness::Partial,
                &raw mut out_spans,
                &raw mut out_error,
            )
        };

        chat_session_classify_status_to_result(status, out_spans, out_error, completeness)
    }
}

impl Drop for ChatSession<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_session_free(self.session.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ffi::CString;
    use std::ptr;

    use serde_json::json;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::chat_session_classify_status_to_result;
    use super::chat_session_create_status_to_result;
    use super::chat_session_parse_status_to_result;
    use super::messages_json;
    use crate::chat_assistant_prefill::ChatAssistantPrefill;
    use crate::chat_assistant_turn::ChatAssistantTurn;
    use crate::chat_classification::ChatClassification;
    use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
    use crate::chat_output::ChatOutput;
    use crate::chat_output_completeness::ChatOutputCompleteness;
    use crate::error::chat_session_error::ChatSessionError;
    use crate::error::parse_chat_message_error::ParseChatMessageError;
    use crate::model::llama_chat_message::LlamaChatMessage;
    use crate::raw_chat_message::RawChatMessage;

    type CreateRefusal = (
        llama_cpp_bindings_sys::llama_rs_chat_session_create_status,
        fn(String) -> ChatSessionError,
    );

    fn parsed_messages(messages_json: &CStr) -> serde_json::Value {
        serde_json::from_str(messages_json.to_str().unwrap()).unwrap()
    }

    #[test]
    fn creating_a_session_reports_each_null_argument_as_a_contract_violation() {
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
        let refusals: [CreateRefusal; 6] = [
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_NOT_JSON,
                |message| ChatSessionError::MessagesNotJson { message },
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_INVALID,
                |message| ChatSessionError::MessagesInvalid { message },
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_NOT_JSON,
                |message| ChatSessionError::ToolsNotJson { message },
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_INVALID,
                |message| ChatSessionError::ToolsInvalid { message },
            ),
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
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_NOT_JSON,
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_INVALID,
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_NOT_JSON,
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_INVALID,
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
        ];

        assert_eq!(
            refusals.map(|status| {
                chat_session_create_status_to_result(status, ptr::null_mut(), ptr::null_mut()).err()
            }),
            refusals.map(|_status| {
                Some(ChatSessionError::FfiContract(FfiContractError {
                    operation: "llama_rs_chat_session_create",
                    detail: "reported a failure without an error message",
                }))
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
    fn parsing_an_answer_that_does_not_match_the_format_keeps_the_raw_answer() {
        let output = ChatOutput::from("not the format");

        assert_eq!(
            chat_session_parse_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_OUTPUT_DOES_NOT_MATCH_FORMAT,
                ptr::null_mut(),
                unsafe {
                    llama_cpp_bindings_sys::llama_rs_string_dup(c"the format differs".as_ptr())
                },
                &output,
                ChatOutputCompleteness::Finished,
            ),
            Ok(ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                text: "not the format".to_owned(),
                completeness: ChatOutputCompleteness::Finished,
                ffi_error_message: "the format differs".to_owned(),
            }))
        );
    }

    #[test]
    fn parsing_an_answer_maps_every_failure_status() {
        let output = ChatOutput::default();
        let parse = |status, out_error| {
            chat_session_parse_status_to_result(
                status,
                ptr::null_mut(),
                out_error,
                &output,
                ChatOutputCompleteness::Partial,
            )
            .err()
        };
        let contract = |detail| {
            Some(ParseChatMessageError::FfiContract(FfiContractError {
                operation: "llama_rs_chat_session_parse",
                detail,
            }))
        };

        assert_eq!(
            [
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_OK,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"the parser threw".as_ptr()) }
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_SESSION_ARG,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_TEXT_ARG,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_BYTE_TOKENS_ARG,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_HANDLE_ARG,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_ERROR_ARG,
                    ptr::null_mut()
                ),
                parse(255, ptr::null_mut()),
            ],
            [
                contract("success status contained a null handle"),
                Some(ParseChatMessageError::Reported {
                    message: "the parser threw".to_owned()
                }),
                Some(ParseChatMessageError::NotEnoughMemory),
                Some(ParseChatMessageError::LlamaCppOutOfMemory),
                contract("was given a null session argument"),
                contract("was given a null text argument"),
                contract("was given a null byte_tokens argument"),
                contract("was given a null out_handle argument"),
                contract("was given a null out_error argument"),
                Some(ParseChatMessageError::FfiStatus(FfiStatusError { operation: "llama_rs_chat_session_parse", code: 255 })),
            ]
        );
    }

    #[test]
    fn classifying_an_answer_that_does_not_match_the_format_is_not_an_error() {
        assert_eq!(
            chat_session_classify_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_OUTPUT_DOES_NOT_MATCH_FORMAT,
                ptr::null_mut(),
                ptr::null_mut(),
                ChatOutputCompleteness::Partial,
            ),
            Ok(ChatClassification::OutputDoesNotMatchFormat(
                ChatOutputCompleteness::Partial
            ))
        );
    }

    #[test]
    fn classifying_an_answer_maps_every_failure_status() {
        let classify = |status, out_error| {
            chat_session_classify_status_to_result(
                status,
                ptr::null_mut(),
                out_error,
                ChatOutputCompleteness::Partial,
            )
            .err()
        };
        let contract = |detail| {
            Some(ParseChatMessageError::FfiContract(FfiContractError {
                operation: "llama_rs_chat_session_classify",
                detail,
            }))
        };

        assert_eq!(
            [
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_OK,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"the parser threw".as_ptr()) }
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_SESSION_ARG,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_TEXT_ARG,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_BYTE_TOKENS_ARG,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_SPANS_ARG,
                    ptr::null_mut()
                ),
                classify(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_ERROR_ARG,
                    ptr::null_mut()
                ),
                classify(255, ptr::null_mut()),
            ],
            [
                contract("success status contained null spans"),
                Some(ParseChatMessageError::Reported {
                    message: "the parser threw".to_owned()
                }),
                Some(ParseChatMessageError::NotEnoughMemory),
                Some(ParseChatMessageError::LlamaCppOutOfMemory),
                contract("was given a null session argument"),
                contract("was given a null text argument"),
                contract("was given a null byte_tokens argument"),
                contract("was given a null out_spans argument"),
                contract("was given a null out_error argument"),
                Some(ParseChatMessageError::FfiStatus(FfiStatusError { operation: "llama_rs_chat_session_classify", code: 255 })),
            ]
        );
    }

    #[test]
    fn a_new_assistant_turn_adds_no_message() {
        let messages = [LlamaChatMessage::new("user".to_owned(), "hi".to_owned()).unwrap()];

        assert_eq!(
            parsed_messages(&messages_json(&messages, &ChatAssistantTurn::New).unwrap()),
            json!([{"role": "user", "content": "hi"}])
        );
    }

    #[test]
    fn a_message_whose_role_or_content_is_not_utf8_is_refused() {
        let not_utf8 = CString::new([0xff]).unwrap();
        let utf8_error = not_utf8.to_str().err();
        let messages = [
            LlamaChatMessage {
                role: not_utf8.clone(),
                content: c"hi".to_owned(),
            },
            LlamaChatMessage {
                role: c"user".to_owned(),
                content: not_utf8,
            },
        ];

        assert_eq!(
            messages.map(|message| messages_json(&[message], &ChatAssistantTurn::New).err()),
            [utf8_error, utf8_error].map(|error| error.map(ChatSessionError::MessageNotUtf8))
        );
    }

    #[test]
    fn continuing_reasoning_appends_an_assistant_message_with_that_reasoning() {
        assert_eq!(
            parsed_messages(
                &messages_json(&[], &ChatAssistantTurn::ContinueReasoning("hmm".to_owned()))
                    .unwrap()
            ),
            json!([{"role": "assistant", "content": "", "reasoning_content": "hmm"}])
        );
    }

    #[test]
    fn continuing_content_appends_an_assistant_message_with_the_prefill() {
        assert_eq!(
            parsed_messages(
                &messages_json(
                    &[],
                    &ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
                        reasoning_content: "done".to_owned(),
                        content: "The".to_owned(),
                    })
                )
                .unwrap()
            ),
            json!([{"role": "assistant", "content": "The", "reasoning_content": "done"}])
        );
    }

    #[test]
    fn a_parse_or_classification_failure_without_a_message_is_a_contract_violation() {
        let output = ChatOutput::default();

        assert_eq!(
            [
                chat_session_parse_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_OUTPUT_DOES_NOT_MATCH_FORMAT,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &output,
                    ChatOutputCompleteness::Finished,
                )
                .err(),
                chat_session_parse_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &output,
                    ChatOutputCompleteness::Finished,
                )
                .err(),
                chat_session_classify_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ChatOutputCompleteness::Finished,
                )
                .err(),
            ],
            [
                Some(ParseChatMessageError::FfiContract(FfiContractError {
                    operation: "llama_rs_chat_session_parse",
                    detail: "reported a failure without an error message",
                })),
                Some(ParseChatMessageError::FfiContract(FfiContractError {
                    operation: "llama_rs_chat_session_parse",
                    detail: "reported a failure without an error message",
                })),
                Some(ParseChatMessageError::FfiContract(FfiContractError {
                    operation: "llama_rs_chat_session_classify",
                    detail: "reported a thrown C++ exception without an error message",
                })),
            ]
        );
    }
}
