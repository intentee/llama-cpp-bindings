use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::llama_chat_template::LlamaChatTemplate;
use crate::model_error::ModelError;

fn chat_templates_init_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_templates_init_status,
    out_templates: llama_cpp_bindings_sys::llama_rs_chat_templates_handle,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_rs_chat_templates>, ModelError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_OK => NonNull::new(out_templates)
            .ok_or_else(|| {
                FfiError::ContractViolated {
                    operation: "llama_rs_chat_templates_init",
                    detail: "success status contained a null templates handle",
                }
                .into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE => {
            Err(ModelError::ModelHasNoChatTemplate)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::ChatTemplatesNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::ChatTemplatesLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_chat_templates_init",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(ModelError::ChatTemplatesReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_MODEL_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_chat_templates_init",
                detail: "was given a null model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_TEMPLATE_OVERRIDE_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_chat_templates_init",
                detail: "was given a null template_override argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_TEMPLATES_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_chat_templates_init",
                detail: "was given a null out_templates argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_ERROR_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_chat_templates_init",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(FfiError::UnknownStatus {
            operation: "llama_rs_chat_templates_init",
            code: i64::from(other),
        }
        .into()),
    }
}

/// Chat templates as llama.cpp prepares them for a model: with the model's BOS and EOS, the
/// template variant for tool calls, and llama.cpp's template fixes applied.
#[derive(Debug)]
pub struct ChatTemplates {
    templates: NonNull<llama_cpp_bindings_sys::llama_rs_chat_templates>,
}

unsafe impl Send for ChatTemplates {}
unsafe impl Sync for ChatTemplates {}

impl ChatTemplates {
    /// The model's own chat templates.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ModelHasNoChatTemplate`] for a model without a chat template, or
    /// another [`ModelError`] when llama.cpp cannot prepare the templates.
    ///
    /// # Safety
    ///
    /// `model` must point to a model llama.cpp loaded and has not freed.
    pub unsafe fn new(
        model: NonNull<llama_cpp_bindings_sys::llama_model>,
    ) -> Result<Self, ModelError> {
        unsafe { Self::init(model, c"") }
    }

    /// `template` in place of the model's own chat templates, the way llama.cpp serves a model
    /// whose template cannot render a request, such as tool calls.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::EmptyTemplateOverride`] for an empty `template`, or another
    /// [`ModelError`] when llama.cpp cannot prepare the template.
    ///
    /// # Safety
    ///
    /// `model` must point to a model llama.cpp loaded and has not freed.
    pub unsafe fn with_override(
        model: NonNull<llama_cpp_bindings_sys::llama_model>,
        template: &LlamaChatTemplate,
    ) -> Result<Self, ModelError> {
        if template.as_c_str().is_empty() {
            return Err(ModelError::EmptyTemplateOverride);
        }

        unsafe { Self::init(model, template.as_c_str()) }
    }

    /// # Safety
    ///
    /// `model` must point to a model llama.cpp loaded and has not freed.
    unsafe fn init(
        model: NonNull<llama_cpp_bindings_sys::llama_model>,
        template_override: &CStr,
    ) -> Result<Self, ModelError> {
        let mut out_templates: llama_cpp_bindings_sys::llama_rs_chat_templates_handle =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_templates_init(
                model.as_ptr(),
                template_override.as_ptr(),
                &raw mut out_templates,
                &raw mut out_error,
            )
        };

        chat_templates_init_status_to_result(status, out_templates, out_error)
            .map(|templates| Self { templates })
    }

    #[must_use]
    pub const fn as_ptr(&self) -> llama_cpp_bindings_sys::llama_rs_chat_templates_handle {
        self.templates.as_ptr()
    }
}

impl Drop for ChatTemplates {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_rs_chat_templates_free(self.templates.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_ffi_status::ffi_error::FfiError;

    use super::chat_templates_init_status_to_result;
    use crate::model_error::ModelError;

    fn init_error(
        status: llama_cpp_bindings_sys::llama_rs_chat_templates_init_status,
        out_error: *mut c_char,
    ) -> Option<ModelError> {
        chat_templates_init_status_to_result(status, ptr::null_mut(), out_error).err()
    }

    const fn contract(detail: &'static str) -> ModelError {
        ModelError::Ffi(FfiError::ContractViolated {
            operation: "llama_rs_chat_templates_init",
            detail,
        })
    }

    fn missing_message(detail: &'static str) -> ModelError {
        ModelError::Ffi(FfiError::ContractViolated {
            operation: "llama_rs_chat_templates_init",
            detail,
        })
    }

    #[test]
    fn preparing_templates_maps_every_failure_status() {
        assert_eq!(
            [
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_OK,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"the template threw".as_ptr()) }
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_MODEL_ARG,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_TEMPLATE_OVERRIDE_ARG,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_TEMPLATES_ARG,
                    ptr::null_mut()
                ),
                init_error(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_ERROR_ARG,
                    ptr::null_mut()
                ),
                init_error(255, ptr::null_mut()),
            ],
            [
                Some(contract("success status contained a null templates handle")),
                Some(ModelError::ModelHasNoChatTemplate),
                Some(ModelError::ChatTemplatesNotEnoughMemory),
                Some(ModelError::ChatTemplatesLlamaCppOutOfMemory),
                Some(ModelError::ChatTemplatesReported {
                    message: "the template threw".to_owned()
                }),
                Some(contract("was given a null model argument")),
                Some(contract("was given a null template_override argument")),
                Some(contract("was given a null out_templates argument")),
                Some(contract("was given a null out_error argument")),
                Some(ModelError::Ffi(FfiError::UnknownStatus {
                    operation: "llama_rs_chat_templates_init",
                    code: 255,
                })),
            ]
        );
    }

    #[test]
    fn a_preparation_exception_without_a_message_is_a_contract_violation() {
        assert_eq!(
            init_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut()
            ),
            Some(missing_message(
                "reported a thrown C++ exception without an error message"
            ))
        );
    }
}
