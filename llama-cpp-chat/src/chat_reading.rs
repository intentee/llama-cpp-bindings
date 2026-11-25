use std::cell::OnceCell;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_free;
use llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_parse_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

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
    reading: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>,
    completeness: ChatOutputCompleteness,
) -> ChatSpans {
    let view = unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_spans_view(reading.as_ptr())
    };
    let llama_cpp_spans = NonNull::new(view.spans.cast_mut())
        .map_or(&[][..], |first_span| unsafe {
            slice::from_raw_parts(first_span.as_ptr(), view.spans_count)
        });

    let spans = llama_cpp_spans
        .iter()
        .map(
            |&llama_cpp_bindings_sys::bindings::llama_rs_chat_span {
                 kind: llama_cpp_kind,
                 start,
                 end,
                 ..
             }| {
                ChatSpan {
                    kind: ChatSpanKind::from(llama_cpp_kind),
                    bytes: start..end,
                }
            },
        )
        .collect::<Vec<ChatSpan>>();

    ChatSpans::new(view.parsed_length, &spans, completeness)
}

fn expected_format_of(
    reading: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>,
) -> Result<String, ChatError> {
    let mut length = 0;
    let format_name = unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_format_name(
            reading.as_ptr(),
            &raw mut length,
        )
    };

    unsafe { cpp_text_view(format_name, length) }
        .map(str::to_owned)
        .map_err(ChatError::FormatNameNotUtf8)
}

fn classification_of(
    reading: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>,
    output: &ChatOutput,
    completeness: ChatOutputCompleteness,
) -> Result<ChatClassification, ChatError> {
    if unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_matches_format(reading.as_ptr())
    } {
        return Ok(ChatClassification::Spans(spans_of(reading, completeness)));
    }

    Ok(ChatClassification::OutputDoesNotMatchFormat(
        RawChatMessage {
            text: output.text().to_owned(),
            completeness,
            expected_format: expected_format_of(reading)?,
        },
    ))
}

/// # Safety
///
/// On `LLAMA_RS_CHAT_READING_PARSE_OK` `out_handle` must be the non-null handle the C++ side
/// created, and on `LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be
/// the non-null message it allocated; the message is freed here.
unsafe fn chat_reading_parse_status_to_result(
    status: llama_rs_chat_reading_parse_status,
    out_handle: *mut llama_cpp_bindings_sys::bindings::llama_rs_parsed_chat,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_rs_parsed_chat>, ChatError> {
    match status {
        llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_handle) })
        }
        llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_chat_reading_parse", |message| {
                    ChatError::ParseReported { message }
                }))
        }
        llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ChatError::ParseNotEnoughMemory)
        }
        llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ChatError::ParseLlamaCppOutOfMemory)
        }
    }
}

fn parsed_message_of(
    reading: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>,
) -> Result<ChatMessageParseOutcome, ChatError> {
    let mut out_handle: *mut llama_cpp_bindings_sys::bindings::llama_rs_parsed_chat =
        ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_parse(
            reading.as_ptr(),
            &raw mut out_handle,
            &raw mut out_error,
        )
    };

    unsafe { chat_reading_parse_status_to_result(status, out_handle, out_error) }
        .and_then(|parsed_chat| unsafe { take_parsed_chat_message(parsed_chat) })
        .map(ChatMessageParseOutcome::Recognized)
}

/// One reading of the model's answer by the template's parser.
///
/// It classifies the answer's bytes right away and maps the same parse into the parsed message
/// the first time it is asked for.
pub struct ChatReading {
    reading: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>,
    classification: ChatClassification,
    parse_outcome: OnceCell<Result<ChatMessageParseOutcome, ChatError>>,
}

unsafe impl Send for ChatReading {}

impl ChatReading {
    /// Takes ownership of `reading`, the parse of `output`, and classifies it.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when the expected format's name is not UTF-8; the reading is
    /// released either way.
    ///
    /// # Safety
    ///
    /// `reading` must point to a reading that `llama_rs_chat_session_read` created and that
    /// nothing else uses or frees: this takes ownership of it, and the returned [`ChatReading`]
    /// frees it when dropped, or this frees it before returning an error.
    pub unsafe fn take(
        reading: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_reading>,
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
                unsafe { llama_rs_chat_reading_free(reading.as_ptr()) };

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
        unsafe { llama_rs_chat_reading_free(self.reading.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_chat_reading_parse_status;

    use super::chat_reading_parse_status_to_result;
    use crate::chat_error::ChatError;

    fn parse_error(
        status: llama_rs_chat_reading_parse_status,
        out_error: *mut c_char,
    ) -> Option<ChatError> {
        unsafe { chat_reading_parse_status_to_result(status, ptr::null_mut(), out_error) }.err()
    }

    #[test]
    fn parsing_a_reading_maps_every_failure_status() {
        assert_eq!(
            [
                parse_error(
                    llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"the mapper threw".as_ptr()) }
                ),
                parse_error(
                    llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                parse_error(
                    llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
            ],
            [
                Some(ChatError::ParseReported {
                    message: "the mapper threw".to_owned()
                }),
                Some(ChatError::ParseNotEnoughMemory),
                Some(ChatError::ParseLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn parsing_a_reading_reports_an_exception_message_that_is_not_utf8() {
        assert_eq!(
            parse_error(
                llama_rs_chat_reading_parse_status::LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"\xff".as_ptr()) }
            ),
            Some(ChatError::ExceptionMessageNotUtf8 {
                operation: "llama_rs_chat_reading_parse",
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            })
        );
    }
}
