use std::cell::OnceCell;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::chat_classification::ChatClassification;
use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
use crate::chat_output::ChatOutput;
use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_span::ChatSpan;
use crate::chat_span_kind::ChatSpanKind;
use crate::chat_spans::ChatSpans;
use crate::cpp_text_view::cpp_text_view;
use crate::error::parse_chat_message_error::ParseChatMessageError;
use crate::parsed_chat::take_parsed_chat_message;
use crate::raw_chat_message::RawChatMessage;

const fn contract_violation(operation: &'static str, detail: &'static str) -> FfiContractError {
    FfiContractError { operation, detail }
}

fn spans_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
    completeness: ChatOutputCompleteness,
) -> Result<ChatSpans, ParseChatMessageError> {
    let view =
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_reading_spans_view(reading.as_ptr()) };
    let wrapper_spans = NonNull::new(view.spans.cast_mut()).map_or(&[][..], |first_span| unsafe {
        slice::from_raw_parts(first_span.as_ptr(), view.spans_count)
    });

    wrapper_spans
        .iter()
        .map(|wrapper_span| {
            ChatSpanKind::try_from(wrapper_span.kind).map(|kind| ChatSpan {
                kind,
                bytes: wrapper_span.start..wrapper_span.end,
            })
        })
        .collect::<Result<Vec<ChatSpan>, FfiStatusError>>()
        .map(|spans| ChatSpans::new(view.parsed_length, &spans, completeness))
        .map_err(ParseChatMessageError::from)
}

fn expected_format_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
) -> Result<String, ParseChatMessageError> {
    let mut length = 0;
    let format_name = unsafe {
        llama_cpp_bindings_sys::llama_rs_chat_reading_format_name(reading.as_ptr(), &raw mut length)
    };

    unsafe { cpp_text_view(format_name, length) }
        .map(str::to_owned)
        .map_err(ParseChatMessageError::FormatNameNotUtf8)
}

fn classification_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
    output: &ChatOutput,
    completeness: ChatOutputCompleteness,
) -> Result<ChatClassification, ParseChatMessageError> {
    if unsafe { llama_cpp_bindings_sys::llama_rs_chat_reading_matches_format(reading.as_ptr()) } {
        return spans_of(reading, completeness).map(ChatClassification::Spans);
    }

    Ok(ChatClassification::OutputDoesNotMatchFormat(
        RawChatMessage {
            text: output.text().to_owned(),
            completeness,
            expected_format: expected_format_of(reading)?,
        },
    ))
}

fn chat_reading_parse_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_reading_parse_status,
    out_handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_rs_parsed_chat>, ParseChatMessageError> {
    const OPERATION: &str = "llama_rs_chat_reading_parse";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_OK => NonNull::new(out_handle)
            .ok_or_else(|| {
                contract_violation(OPERATION, "success status contained a null handle").into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
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
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_READING_ARG => {
            Err(contract_violation(OPERATION, "was given a null reading argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_HANDLE_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_handle argument").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_ERROR_ARG => {
            Err(contract_violation(OPERATION, "was given a null out_error argument").into())
        }
        other => Err(FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

fn parsed_message_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
    let mut out_handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_chat_reading_parse(
            reading.as_ptr(),
            &raw mut out_handle,
            &raw mut out_error,
        )
    };

    chat_reading_parse_status_to_result(status, out_handle, out_error)
        .and_then(take_parsed_chat_message)
        .map(ChatMessageParseOutcome::Recognized)
}

/// One reading of the model's answer by the template's parser.
///
/// It classifies the answer's bytes right away and builds the parsed message the first time it
/// is asked for: a partial answer's message comes from the same parse, and a finished answer's
/// message comes from llama.cpp completing it as llama.cpp completes every finished message.
pub struct ChatReading {
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
    classification: ChatClassification,
    parse_outcome: OnceCell<Result<ChatMessageParseOutcome, ParseChatMessageError>>,
}

unsafe impl Send for ChatReading {}

impl ChatReading {
    /// Takes ownership of `reading`, the parse of `output`, and classifies it.
    ///
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when a span kind is unknown to this binding or the
    /// expected format's name is not UTF-8; the reading is released either way.
    pub fn take(
        reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<Self, ParseChatMessageError> {
        match classification_of(reading, output, completeness) {
            Ok(classification) => Ok(Self {
                reading,
                classification,
                parse_outcome: OnceCell::new(),
            }),
            Err(classification_failure) => {
                unsafe { llama_cpp_bindings_sys::llama_rs_chat_reading_free(reading.as_ptr()) };

                Err(classification_failure)
            }
        }
    }

    #[must_use]
    pub const fn classification(&self) -> &ChatClassification {
        &self.classification
    }

    /// The message this reading holds, or the raw answer the parser did not recognize, built
    /// once and kept for every later call.
    ///
    /// # Errors
    ///
    /// Returns the [`ParseChatMessageError`] of llama.cpp failing while mapping the parse into a
    /// message, or reporting text that is not UTF-8.
    pub fn parse(&self) -> Result<&ChatMessageParseOutcome, &ParseChatMessageError> {
        self.parse_outcome
            .get_or_init(|| match &self.classification {
                ChatClassification::Spans(_) => parsed_message_of(self.reading),
                ChatClassification::OutputDoesNotMatchFormat(raw_chat_message) => Ok(
                    ChatMessageParseOutcome::Unrecognized(raw_chat_message.clone()),
                ),
            })
            .as_ref()
    }
}

impl Drop for ChatReading {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_reading_free(self.reading.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::chat_reading_parse_status_to_result;
    use crate::error::parse_chat_message_error::ParseChatMessageError;

    #[test]
    fn parsing_a_reading_maps_every_failure_status() {
        let parse = |status, out_error| {
            chat_reading_parse_status_to_result(status, ptr::null_mut(), out_error).err()
        };
        let contract = |detail| {
            Some(ParseChatMessageError::FfiContract(FfiContractError {
                operation: "llama_rs_chat_reading_parse",
                detail,
            }))
        };

        assert_eq!(
            [
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_OK,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"the mapper threw".as_ptr()) }
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_READING_ARG,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_HANDLE_ARG,
                    ptr::null_mut()
                ),
                parse(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_ERROR_ARG,
                    ptr::null_mut()
                ),
                parse(255, ptr::null_mut()),
            ],
            [
                contract("success status contained a null handle"),
                Some(ParseChatMessageError::Reported {
                    message: "the mapper threw".to_owned()
                }),
                Some(ParseChatMessageError::CppString(CppStringError::Missing(FfiContractError {
                    operation: "llama_rs_chat_reading_parse",
                    detail: "reported a thrown C++ exception without an error message",
                }))),
                Some(ParseChatMessageError::NotEnoughMemory),
                Some(ParseChatMessageError::LlamaCppOutOfMemory),
                contract("was given a null reading argument"),
                contract("was given a null out_handle argument"),
                contract("was given a null out_error argument"),
                Some(ParseChatMessageError::FfiStatus(FfiStatusError {
                    operation: "llama_rs_chat_reading_parse",
                    code: 255
                })),
            ]
        );
    }
}
