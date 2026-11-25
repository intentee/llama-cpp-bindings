use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;
use std::sync::Arc;

use llama_cpp_bindings_sys::bindings::llama_rs_decision_model_init_status;
use llama_cpp_bindings_sys::bindings::llama_rs_decision_model_output;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;
use llama_cpp_model::llama_model::LlamaModel;

use crate::decision_error::DecisionError;
use crate::decision_model_handle::DecisionModelHandle;
use crate::decision_model_output::DecisionModelOutput;

unsafe fn decision_model_init_status_to_result(
    status: llama_rs_decision_model_init_status,
    out_model: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_model,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_rs_decision_model>, DecisionError> {
    match status {
        llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_model) })
        }
        llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_MODEL_IS_NOT_A_DECISION_MODEL => {
            Err(DecisionError::ModelIsNotADecisionModel)
        }
        llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_DECISION_TYPE_UNSUPPORTED_BY_LLAMA_CPP => {
            Err(DecisionError::DecisionTypeUnsupportedByLlamaCpp)
        }
        llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(DecisionError::ModelNotEnoughMemory)
        }
        llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(DecisionError::ModelLlamaCppOutOfMemory)
        }
        llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_REJECTED_MODEL => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_decision_model_init", |message| {
                    DecisionError::ModelRejected { message }
                }))
        }
    }
}

#[derive(Clone)]
pub struct DecisionModel {
    pub model: Arc<DecisionModelHandle>,
    pub has_causal_attention: bool,
}

impl DecisionModel {
    /// # Errors
    ///
    /// Returns [`DecisionError::ModelIsNotADecisionModel`] for a model that declares no decision
    /// type, [`DecisionError::DecisionTypeUnsupportedByLlamaCpp`] for a decision type llama.cpp
    /// does not know, and [`DecisionError::ModelRejected`] when llama.cpp cannot serve decisions
    /// of the model.
    pub fn new(model: Arc<LlamaModel>) -> Result<Self, DecisionError> {
        let mut out_model: *mut llama_cpp_bindings_sys::bindings::llama_rs_decision_model =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_model_init(
                model.as_ptr(),
                &raw mut out_model,
                &raw mut out_error,
            )
        };
        let handle = unsafe { decision_model_init_status_to_result(status, out_model, out_error) }?;
        let has_causal_attention =
            unsafe { llama_cpp_bindings_sys::bindings::llama_rs_model_causal_attn(model.as_ptr()) };

        Ok(Self {
            model: Arc::new(DecisionModelHandle {
                handle,
                llama_model: model,
            }),
            has_causal_attention,
        })
    }

    #[must_use]
    pub fn can_share_prompt(&self) -> bool {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_model_can_share_prompt(
                self.model.handle.as_ptr(),
            )
        }
    }

    #[must_use]
    pub fn can_use_images(&self) -> bool {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_model_can_use_images(
                self.model.handle.as_ptr(),
            )
        }
    }

    #[must_use]
    pub fn is_joint(&self) -> bool {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_model_is_joint(
                self.model.handle.as_ptr(),
            )
        }
    }

    /// The media marker llama.cpp lays decision prompts out with. A projector that tokenizes
    /// decision prompts with media has to be created with it.
    #[must_use]
    pub fn media_marker() -> &'static CStr {
        unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::bindings::llama_rs_decision_media_marker())
        }
    }

    #[must_use]
    pub fn output(&self) -> DecisionModelOutput {
        match unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_model_output_of(
                self.model.handle.as_ptr(),
            )
        } {
            llama_rs_decision_model_output::LLAMA_RS_DECISION_MODEL_OUTPUT_LOGITS => {
                DecisionModelOutput::Logits
            }
            llama_rs_decision_model_output::LLAMA_RS_DECISION_MODEL_OUTPUT_TOKEN_EMBEDDINGS => {
                DecisionModelOutput::TokenEmbeddings
            }
            llama_rs_decision_model_output::LLAMA_RS_DECISION_MODEL_OUTPUT_POOLED_EMBEDDINGS => {
                DecisionModelOutput::PooledEmbeddings
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_decision_model_init_status;

    use super::decision_model_init_status_to_result;
    use crate::decision_error::DecisionError;

    fn init_error(
        status: llama_rs_decision_model_init_status,
        out_error: *mut c_char,
    ) -> Option<DecisionError> {
        unsafe { decision_model_init_status_to_result(status, ptr::null_mut(), out_error) }.err()
    }

    #[test]
    fn a_decision_model_llama_cpp_runs_out_of_memory_for_reports_it() {
        let failures = [
            init_error(
                llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            init_error(
                llama_rs_decision_model_init_status::LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
            ),
        ];

        assert!(matches!(
            failures,
            [
                Some(DecisionError::ModelNotEnoughMemory),
                Some(DecisionError::ModelLlamaCppOutOfMemory),
            ]
        ));
    }
}
