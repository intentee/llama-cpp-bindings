use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use indexmap::IndexMap;
use llama_cpp_bindings_sys::bindings::llama_rs_decision_answers_read_status;
use llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_fill_status;
use llama_cpp_bindings_sys::bindings::llama_rs_decision_prompts_share_status;
use llama_cpp_bindings_sys::bindings::llama_rs_decision_request_parse_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;
use llama_cpp_bindings_types::decision_answer::DecisionAnswer;
use llama_cpp_bindings_types::decision_request::DecisionRequest;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use serde_json::Value;

use crate::decision_error::DecisionError;
use crate::decision_model::DecisionModel;
use crate::decision_prompt::DecisionPrompt;
use crate::decision_prompt_handle::DecisionPromptHandle;
use crate::decision_prompt_outputs::DecisionPromptOutputs;
use crate::decision_prompts::DecisionPrompts;
use crate::decision_request_handle::DecisionRequestHandle;

unsafe fn request_parse_status_to_result(
    status: llama_rs_decision_request_parse_status,
    out_request: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_request,
    out_error: *mut c_char,
) -> Result<DecisionRequestHandle, DecisionError> {
    match status {
        llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_OK => {
            Ok(DecisionRequestHandle {
                handle: unsafe { NonNull::new_unchecked(out_request) },
            })
        }
        llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_BODY_IS_NOT_JSON => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_request_parse", |message| {
                    DecisionError::RequestBodyIsNotJson { message }
                }))
        }
        llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_REQUEST_REJECTED => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_request_parse", |message| {
                    DecisionError::RequestRejected { message }
                }))
        }
        llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(DecisionError::RequestNotEnoughMemory)
        }
        llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(DecisionError::RequestLlamaCppOutOfMemory)
        }
        llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_request_parse", |message| {
                    DecisionError::RequestReported { message }
                }))
        }
    }
}

unsafe fn prompt_fill_status_to_result(
    status: llama_rs_decision_prompt_fill_status,
    out_prompt: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt,
    out_error: *mut c_char,
) -> Result<DecisionPromptHandle, DecisionError> {
    match status {
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_OK => {
            Ok(DecisionPromptHandle {
                handle: unsafe { NonNull::new_unchecked(out_prompt) },
            })
        }
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_UNSUPPORTED_BY_DECISION_MODEL => {
            Err(DecisionError::MediaUnsupportedByDecisionModel)
        }
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_REQUIRES_A_PROJECTOR => {
            Err(DecisionError::MediaRequiresAProjector)
        }
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_REQUEST_REJECTED => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_prompt_fill", |message| {
                    DecisionError::PromptRejected { message }
                }))
        }
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_ERROR_STRING_ALLOCATION_FAILED => {
            Err(DecisionError::PromptNotEnoughMemory)
        }
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(DecisionError::PromptLlamaCppOutOfMemory)
        }
        llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_prompt_fill", |message| {
                    DecisionError::PromptReported { message }
                }))
        }
    }
}

unsafe fn answers_read_status_to_result(
    status: llama_rs_decision_answers_read_status,
    out_answers_json: *mut c_char,
    out_error: *mut c_char,
) -> Result<IndexMap<String, DecisionAnswer>, DecisionError> {
    match status {
        llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_OK => {
            unsafe { CppAllocatedString::from_raw(out_answers_json) }
                .into_string()
                .map_err(DecisionError::AnswersTextNotUtf8)
                .and_then(|answers_json| {
                    serde_json::from_str(&answers_json)
                        .map_err(DecisionError::AnswersAreNotDecisionAnswers)
                })
        }
        llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_OUTPUTS_UNREADABLE => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_request_read_answers", |message| {
                    DecisionError::OutputsUnreadable { message }
                }))
        }
        llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_SCORES_REJECTED => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_request_read_answers", |message| {
                    DecisionError::AnswersScoresRejected { message }
                }))
        }
        llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED => {
            Err(DecisionError::AnswersNotEnoughMemory)
        }
        llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(DecisionError::AnswersLlamaCppOutOfMemory)
        }
        llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_request_read_answers", |message| {
                    DecisionError::AnswersReported { message }
                }))
        }
    }
}

const fn prompts_share_status_to_result(
    status: llama_rs_decision_prompts_share_status,
    shared_prefix_length: usize,
) -> Result<usize, DecisionError> {
    match status {
        llama_rs_decision_prompts_share_status::LLAMA_RS_DECISION_PROMPTS_SHARE_OK => {
            Ok(shared_prefix_length)
        }
        llama_rs_decision_prompts_share_status::LLAMA_RS_DECISION_PROMPTS_SHARE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(DecisionError::PromptSharingLlamaCppOutOfMemory)
        }
    }
}

fn shared_prefix_length_of(prompts: &[DecisionPrompt]) -> Result<usize, DecisionError> {
    let mut handles: Vec<*mut llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt> = prompts
        .iter()
        .map(|prompt| prompt.prompt.handle.as_ptr())
        .collect();
    let mut shared_prefix_length = 0;
    let status = unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_decision_prompts_shared_prefix(
            handles.as_mut_ptr(),
            handles.len(),
            &raw mut shared_prefix_length,
        )
    };

    prompts_share_status_to_result(status, shared_prefix_length)
}

fn projector_pointer(
    projector: Option<&MtmdContext>,
) -> *mut llama_cpp_bindings_sys::bindings::mtmd_context {
    projector.map_or(ptr::null_mut(), MtmdContext::as_ptr)
}

/// A decision request as llama.cpp read it for one decision model: the prompts that answer its
/// questions, and the answers it formats from their scores.
pub struct PreparedDecisionRequest {
    pub decision_model: DecisionModel,
    pub has_media: bool,
    pub request: DecisionRequestHandle,
}

impl PreparedDecisionRequest {
    /// # Errors
    ///
    /// Returns [`DecisionError::RequestRejected`] when llama.cpp refuses the request for this
    /// decision model, with llama.cpp's reason.
    pub fn new(
        decision_model: &DecisionModel,
        request: &DecisionRequest,
    ) -> Result<Self, DecisionError> {
        let body = Value::from(request).to_string();
        let mut out_request: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_request =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_request_parse(
                decision_model.model.handle.as_ptr(),
                body.as_ptr().cast::<c_char>(),
                body.len(),
                &raw mut out_request,
                &raw mut out_error,
            )
        };
        let request = unsafe { request_parse_status_to_result(status, out_request, out_error) }?;

        Ok(Self {
            decision_model: decision_model.clone(),
            has_media: unsafe {
                llama_cpp_bindings_sys::bindings::llama_rs_decision_request_has_media(
                    request.handle.as_ptr(),
                )
            },
            request,
        })
    }

    /// The prompts that answer every question, and the leading tokens llama.cpp lets them share.
    ///
    /// # Errors
    ///
    /// Returns [`DecisionError::PromptRejected`] when llama.cpp cannot lay out a prompt for the
    /// request, [`DecisionError::MediaRequiresAProjector`] when the request carries media and
    /// `projector` is `None`, and [`DecisionError::PromptSharingLlamaCppOutOfMemory`] when
    /// llama.cpp runs out of memory finding the tokens the prompts share.
    pub fn prompts(
        &self,
        projector: Option<&MtmdContext>,
    ) -> Result<DecisionPrompts, DecisionError> {
        if self.decision_model.is_joint() {
            return Ok(DecisionPrompts {
                prompts: vec![self.joint_prompt(projector)?],
                shared_prefix_length: 0,
            });
        }

        let question_count = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_request_question_count(
                self.request.handle.as_ptr(),
            )
        };
        let prompts = (0..question_count)
            .flat_map(|question_index| {
                (0..self.variant_count(question_index))
                    .map(move |variant| self.question_prompt(question_index, variant, projector))
            })
            .collect::<Result<Vec<DecisionPrompt>, DecisionError>>()?;
        let shared_prefix_length = if self.decision_model.shares_prompt_prefix() {
            shared_prefix_length_of(&prompts)?
        } else {
            0
        };

        Ok(DecisionPrompts {
            prompts,
            shared_prefix_length,
        })
    }

    /// Reads the scores of `prompt` from `context`, whose final batch decoded the prompt's last
    /// tokens as outputs at `outputs`, and returns the answers of the questions whose every
    /// prompt is now read, by question id, the way llama.cpp formats decision answers.
    ///
    /// # Errors
    ///
    /// Returns [`DecisionError::OutputsUnreadable`] when the final batch holds no output the
    /// prompt is read from, and [`DecisionError::AnswersScoresRejected`] when llama.cpp cannot
    /// format an answer from the scores.
    pub fn read_answers(
        &mut self,
        prompt: &DecisionPrompt,
        context: &LlamaContext<'_>,
        DecisionPromptOutputs {
            batch_indices,
            sequence_id,
        }: DecisionPromptOutputs<'_>,
    ) -> Result<IndexMap<String, DecisionAnswer>, DecisionError> {
        let mut out_answers_json: *mut c_char = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_request_read_answers(
                self.decision_model.model.handle.as_ptr(),
                self.request.handle.as_ptr(),
                prompt.prompt.handle.as_ptr(),
                context.as_ptr(),
                batch_indices.as_ptr(),
                batch_indices.len(),
                sequence_id,
                &raw mut out_answers_json,
                &raw mut out_error,
            )
        };

        unsafe { answers_read_status_to_result(status, out_answers_json, out_error) }
    }

    fn joint_prompt(
        &self,
        projector: Option<&MtmdContext>,
    ) -> Result<DecisionPrompt, DecisionError> {
        let mut out_prompt: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_fill_joint(
                self.decision_model.model.handle.as_ptr(),
                self.request.handle.as_ptr(),
                projector_pointer(projector),
                &raw mut out_prompt,
                &raw mut out_error,
            )
        };

        Ok(DecisionPrompt {
            decision_model: self.decision_model.clone(),
            prompt: unsafe { prompt_fill_status_to_result(status, out_prompt, out_error) }?,
        })
    }

    fn question_prompt(
        &self,
        question_index: usize,
        variant: usize,
        projector: Option<&MtmdContext>,
    ) -> Result<DecisionPrompt, DecisionError> {
        let mut out_prompt: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_fill(
                self.decision_model.model.handle.as_ptr(),
                self.request.handle.as_ptr(),
                question_index,
                variant,
                projector_pointer(projector),
                &raw mut out_prompt,
                &raw mut out_error,
            )
        };

        Ok(DecisionPrompt {
            decision_model: self.decision_model.clone(),
            prompt: unsafe { prompt_fill_status_to_result(status, out_prompt, out_error) }?,
        })
    }

    fn variant_count(&self, question_index: usize) -> usize {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_request_question_variant_count(
                self.decision_model.model.handle.as_ptr(),
                self.request.handle.as_ptr(),
                question_index,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_decision_answers_read_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_fill_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_decision_prompts_share_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_decision_request_parse_status;

    use super::answers_read_status_to_result;
    use super::prompt_fill_status_to_result;
    use super::prompts_share_status_to_result;
    use super::request_parse_status_to_result;
    use crate::decision_error::DecisionError;

    fn duplicated(text: &CStr) -> *mut c_char {
        unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(text.as_ptr()) }
    }

    #[test]
    fn a_request_llama_cpp_cannot_read_reports_why() {
        let failures = [
            unsafe {
                request_parse_status_to_result(
                    llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_BODY_IS_NOT_JSON,
                    ptr::null_mut(),
                    duplicated(c"not json"),
                )
            }
            .err(),
            unsafe {
                request_parse_status_to_result(
                    llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_REQUEST_REJECTED,
                    ptr::null_mut(),
                    duplicated(c"\"questions\" must be a non-empty object"),
                )
            }
            .err(),
            unsafe {
                request_parse_status_to_result(
                    llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
            .err(),
            unsafe {
                request_parse_status_to_result(
                    llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
            .err(),
            unsafe {
                request_parse_status_to_result(
                    llama_rs_decision_request_parse_status::LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    duplicated(c"threw"),
                )
            }
            .err(),
        ];

        assert!(matches!(
            failures,
            [
                Some(DecisionError::RequestBodyIsNotJson { .. }),
                Some(DecisionError::RequestRejected { .. }),
                Some(DecisionError::RequestNotEnoughMemory),
                Some(DecisionError::RequestLlamaCppOutOfMemory),
                Some(DecisionError::RequestReported { .. }),
            ]
        ));
    }

    #[test]
    fn a_prompt_llama_cpp_cannot_lay_out_reports_why() {
        let failures = [
            unsafe {
                prompt_fill_status_to_result(
                    llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_UNSUPPORTED_BY_DECISION_MODEL,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
            .err(),
            unsafe {
                prompt_fill_status_to_result(
                    llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_REQUIRES_A_PROJECTOR,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
            .err(),
            unsafe {
                prompt_fill_status_to_result(
                    llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_REQUEST_REJECTED,
                    ptr::null_mut(),
                    duplicated(c"this model answers exactly one question"),
                )
            }
            .err(),
            unsafe {
                prompt_fill_status_to_result(
                    llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
            .err(),
            unsafe {
                prompt_fill_status_to_result(
                    llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
            .err(),
            unsafe {
                prompt_fill_status_to_result(
                    llama_rs_decision_prompt_fill_status::LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    duplicated(c"threw"),
                )
            }
            .err(),
        ];

        assert!(matches!(
            failures,
            [
                Some(DecisionError::MediaUnsupportedByDecisionModel),
                Some(DecisionError::MediaRequiresAProjector),
                Some(DecisionError::PromptRejected { .. }),
                Some(DecisionError::PromptNotEnoughMemory),
                Some(DecisionError::PromptLlamaCppOutOfMemory),
                Some(DecisionError::PromptReported { .. }),
            ]
        ));
    }

    #[test]
    fn answers_llama_cpp_cannot_read_report_why() {
        let failure = |status, out_answers_json, out_error| {
            unsafe { answers_read_status_to_result(status, out_answers_json, out_error) }.err()
        };
        let failures = [
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_OK,
                duplicated(c"{\"weather\":{\"type\":\"unknown\"}}"),
                ptr::null_mut(),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_OK,
                duplicated(c"\xff"),
                ptr::null_mut(),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_OUTPUTS_UNREADABLE,
                ptr::null_mut(),
                duplicated(c"no outputs"),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_SCORES_REJECTED,
                ptr::null_mut(),
                duplicated(c"the scores hold no probability"),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_SCORES_REJECTED,
                ptr::null_mut(),
                duplicated(c"\xff"),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
            ),
            failure(
                llama_rs_decision_answers_read_status::LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                duplicated(c"threw"),
            ),
        ];

        assert!(matches!(
            failures,
            [
                Some(DecisionError::AnswersAreNotDecisionAnswers(_)),
                Some(DecisionError::AnswersTextNotUtf8(_)),
                Some(DecisionError::OutputsUnreadable { .. }),
                Some(DecisionError::AnswersScoresRejected { .. }),
                Some(DecisionError::ExceptionMessageNotUtf8 { .. }),
                Some(DecisionError::AnswersNotEnoughMemory),
                Some(DecisionError::AnswersLlamaCppOutOfMemory),
                Some(DecisionError::AnswersReported { .. }),
            ]
        ));
    }

    #[test]
    fn prompts_llama_cpp_runs_out_of_memory_sharing_report_it() {
        assert!(matches!(
            prompts_share_status_to_result(
                llama_rs_decision_prompts_share_status::LLAMA_RS_DECISION_PROMPTS_SHARE_LLAMA_CPP_OUT_OF_MEMORY,
                0,
            ),
            Err(DecisionError::PromptSharingLlamaCppOutOfMemory)
        ));
    }
}
