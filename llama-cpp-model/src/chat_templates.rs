use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_init_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::llama_chat_template::LlamaChatTemplate;
use crate::model_error::ModelError;

/// # Safety
///
/// On `LLAMA_RS_CHAT_TEMPLATES_INIT_OK` `out_templates` must be the non-null handle the C++ side
/// created, and on `LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be
/// the non-null message it allocated; the message is freed here.
unsafe fn chat_templates_init_status_to_result(
    status: llama_rs_chat_templates_init_status,
    out_templates: llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_handle,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_templates>, ModelError> {
    match status {
        llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_templates) })
        }
        llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE => {
            Err(ModelError::ModelHasNoChatTemplate)
        }
        llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::ChatTemplatesNotEnoughMemory)
        }
        llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::ChatTemplatesLlamaCppOutOfMemory)
        }
        llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_chat_templates_init", |message| {
                    ModelError::ChatTemplatesReported { message }
                }))
        }
    }
}

/// Chat templates as llama.cpp prepares them for a model: with the model's BOS and EOS, the
/// template variant for tool calls, and llama.cpp's template fixes applied.
#[derive(Debug)]
pub struct ChatTemplates {
    templates: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_chat_templates>,
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
        model: NonNull<llama_cpp_bindings_sys::bindings::llama_model>,
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
        model: NonNull<llama_cpp_bindings_sys::bindings::llama_model>,
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
        model: NonNull<llama_cpp_bindings_sys::bindings::llama_model>,
        template_override: &CStr,
    ) -> Result<Self, ModelError> {
        let mut out_templates: llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_handle =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_init(
                model.as_ptr(),
                template_override.as_ptr(),
                &raw mut out_templates,
                &raw mut out_error,
            )
        };

        unsafe { chat_templates_init_status_to_result(status, out_templates, out_error) }
            .map(|templates| Self { templates })
    }

    #[must_use]
    pub const fn as_ptr(&self) -> llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_handle {
        self.templates.as_ptr()
    }
}

impl Drop for ChatTemplates {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_free(self.templates.as_ptr());
        };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_chat_templates_init_status;

    use super::chat_templates_init_status_to_result;
    use crate::model_error::ModelError;

    fn init_error(
        status: llama_rs_chat_templates_init_status,
        out_error: *mut c_char,
    ) -> Option<ModelError> {
        unsafe { chat_templates_init_status_to_result(status, ptr::null_mut(), out_error) }.err()
    }

    #[test]
    fn preparing_templates_maps_every_failure_status() {
        assert_eq!(
            [
                init_error(
                    llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE,
                    ptr::null_mut()
                ),
                init_error(
                    llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut()
                ),
                init_error(
                    llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut()
                ),
                init_error(
                    llama_rs_chat_templates_init_status::LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"the template threw".as_ptr()) }
                ),
            ],
            [
                Some(ModelError::ModelHasNoChatTemplate),
                Some(ModelError::ChatTemplatesNotEnoughMemory),
                Some(ModelError::ChatTemplatesLlamaCppOutOfMemory),
                Some(ModelError::ChatTemplatesReported {
                    message: "the template threw".to_owned()
                }),
            ]
        );
    }
}
