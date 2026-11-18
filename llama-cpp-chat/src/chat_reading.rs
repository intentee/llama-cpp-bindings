use std::cell::OnceCell;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_wrapper_string::read_and_free_exception_message::read_and_free_exception_message;

use crate::chat_classification::ChatClassification;
use crate::chat_error::ChatError;
use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
use crate::chat_output::ChatOutput;
use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_span::ChatSpan;
use crate::chat_span_kind::ChatSpanKind;
use crate::chat_spans::ChatSpans;
use crate::cpp_text_view::cpp_text_view;
use crate::raw_chat_message::RawChatMessage;
use crate::take_parsed_chat_message::take_parsed_chat_message;

fn spans_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
    completeness: ChatOutputCompleteness,
) -> Result<ChatSpans, ChatError> {
    let view =
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_reading_spans_view(reading.as_ptr()) };
    let wrapper_spans = NonNull::new(view.spans.cast_mut()).map_or(&[][..], |first_span| unsafe {
        slice::from_raw_parts(first_span.as_ptr(), view.spans_count)
    });

    wrapper_spans
        .iter()
        .map(
            |&llama_cpp_bindings_sys::llama_rs_chat_span {
                 kind: wrapper_kind,
                 start,
                 end,
                 ..
             }| {
                ChatSpanKind::try_from(wrapper_kind).map(|kind| ChatSpan {
                    kind,
                    bytes: start..end,
                })
            },
        )
        .collect::<Result<Vec<ChatSpan>, ChatError>>()
        .map(|spans| ChatSpans::new(view.parsed_length, &spans, completeness))
}

fn expected_format_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
) -> Result<String, ChatError> {
    let mut length = 0;
    let format_name = unsafe {
        llama_cpp_bindings_sys::llama_rs_chat_reading_format_name(reading.as_ptr(), &raw mut length)
    };

    unsafe { cpp_text_view(format_name, length) }
        .map(str::to_owned)
        .map_err(ChatError::FormatNameNotUtf8)
}

fn classification_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
    output: &ChatOutput,
    completeness: ChatOutputCompleteness,
) -> Result<ChatClassification, ChatError> {
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
) -> Result<NonNull<llama_cpp_bindings_sys::llama_rs_parsed_chat>, ChatError> {
    const OPERATION: &str = "llama_rs_chat_reading_parse";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_OK => {
            NonNull::new(out_handle).ok_or(ChatError::WrapperSucceededWithoutOutput {
                operation: OPERATION,
                output: "handle",
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(ChatError::ParseReported {
                message: unsafe { read_and_free_exception_message(OPERATION, out_error) }?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ChatError::ParseNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ChatError::ParseLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_READING_ARG => {
            Err(ChatError::WrapperRefusedNullArgument {
                operation: OPERATION,
                argument: "reading",
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_HANDLE_ARG => {
            Err(ChatError::WrapperRefusedNullArgument {
                operation: OPERATION,
                argument: "out_handle",
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_ERROR_ARG => {
            Err(ChatError::WrapperRefusedNullArgument {
                operation: OPERATION,
                argument: "out_error",
            })
        }
        other => Err(ChatError::UnknownWrapperStatus {
            operation: OPERATION,
            code: i64::from(other),
        }),
    }
}

fn parsed_message_of(
    reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
) -> Result<ChatMessageParseOutcome, ChatError> {
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
    parse_outcome: OnceCell<Result<ChatMessageParseOutcome, ChatError>>,
}

unsafe impl Send for ChatReading {}

impl ChatReading {
    /// Takes ownership of `reading`, the parse of `output`, and classifies it.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when a span kind is unknown to this binding or the
    /// expected format's name is not UTF-8; the reading is released either way.
    pub fn take(
        reading: NonNull<llama_cpp_bindings_sys::llama_rs_chat_reading>,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<Self, ChatError> {
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
    /// Returns the [`ChatError`] of llama.cpp failing while mapping the parse into a
    /// message, or reporting text that is not UTF-8.
    pub fn parse(&self) -> Result<&ChatMessageParseOutcome, &ChatError> {
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

    use llama_cpp_wrapper_string::wrapper_string_error::WrapperStringError;

    use super::chat_reading_parse_status_to_result;
    use crate::chat_error::ChatError;

    #[test]
    fn parsing_a_reading_maps_every_failure_status() {
        let parse = |status, out_error| {
            chat_reading_parse_status_to_result(status, ptr::null_mut(), out_error).err()
        };
        let refused_null_argument = |argument| {
            Some(ChatError::WrapperRefusedNullArgument {
                operation: "llama_rs_chat_reading_parse",
                argument,
            })
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
                Some(ChatError::WrapperSucceededWithoutOutput {
                    operation: "llama_rs_chat_reading_parse",
                    output: "handle",
                }),
                Some(ChatError::ParseReported {
                    message: "the mapper threw".to_owned()
                }),
                Some(ChatError::WrapperString(WrapperStringError::ExceptionMessageMissing { operation: "llama_rs_chat_reading_parse" })),
                Some(ChatError::ParseNotEnoughMemory),
                Some(ChatError::ParseLlamaCppOutOfMemory),
                refused_null_argument("reading"),
                refused_null_argument("out_handle"),
                refused_null_argument("out_error"),
                Some(ChatError::UnknownWrapperStatus {
                    operation: "llama_rs_chat_reading_parse",
                    code: 255,
                }),
            ]
        );
    }
}
