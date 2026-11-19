use std::ffi::NulError;
use std::ffi::{CStr, CString, c_char};
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ptr;
use std::str::Utf8Error;

use llama_cpp_bindings_sys::bindings::llama_rs_render_template_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::model_error::ModelError;
use crate::read_exception_message::read_exception_message;

const OPERATION: &str = "llama_rs_render_template";

/// # Safety
///
/// On `LLAMA_RS_RENDER_TEMPLATE_OK` `out_rendered` must be the non-null text the C++ side
/// allocated, and on the statuses that carry a message `out_error` must be the non-null message
/// it allocated; whichever is read is freed here.
unsafe fn render_template_status_to_result(
    status: llama_rs_render_template_status,
    out_rendered: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ModelError> {
    let reported_message = || -> Result<String, ModelError> {
        unsafe { read_exception_message(OPERATION, out_error) }
    };
    match status {
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_OK => {
            unsafe { CppAllocatedString::from_raw(out_rendered) }
                .into_string()
                .map_err(|source| ModelError::RenderedTemplateNotUtf8 { source })
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::RenderNotEnoughMemory)
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::RenderLlamaCppOutOfMemory)
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_INVALID_JSON => {
            Err(ModelError::TemplateGlobalsInvalidJson {
                message: reported_message()?,
            })
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_GLOBALS_NOT_A_JSON_OBJECT => {
            Err(ModelError::TemplateGlobalsNotAJsonObject)
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_INVALID_TEMPLATE => {
            Err(ModelError::TemplateInvalid {
                message: reported_message()?,
            })
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_EXECUTION_FAILED => {
            Err(ModelError::TemplateExecutionFailed {
                message: reported_message()?,
            })
        }
        llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(ModelError::RenderReported {
                message: reported_message()?,
            })
        }
    }
}

#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(pub CString);

impl LlamaChatTemplate {
    /// # Errors
    /// Returns an error if the template string contains null bytes.
    pub fn new(template: &str) -> Result<Self, NulError> {
        Ok(Self(CString::new(template)?))
    }

    #[must_use]
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// # Errors
    ///
    /// Returns [`ModelError`] when `globals_json` contains a NUL byte, is not valid
    /// JSON or not a JSON object, the template cannot be parsed or executed, llama.cpp fails
    /// while rendering, or the rendered text or error message is not valid UTF-8.
    pub fn render(&self, globals_json: &str) -> Result<String, ModelError> {
        let globals =
            CString::new(globals_json).map_err(ModelError::TemplateGlobalsContainNulByte)?;
        let mut out_rendered: *mut c_char = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_render_template(
                self.0.as_ptr(),
                globals.as_ptr(),
                &raw mut out_rendered,
                &raw mut out_error,
            )
        };

        unsafe { render_template_status_to_result(status, out_rendered, out_error) }
    }

    /// # Errors
    /// Returns an error if the template is not valid UTF-8.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// # Errors
    /// Returns an error if the template is not valid UTF-8.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl Debug for LlamaChatTemplate {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_render_template_status;

    use super::render_template_status_to_result;
    use crate::llama_chat_template::LlamaChatTemplate;
    use crate::model_error::ModelError;

    #[test]
    fn a_template_keeps_its_name() {
        let template = LlamaChatTemplate::new("chatml").unwrap();
        let template_str = template.to_str().unwrap();

        assert_eq!(template_str, "chatml");
    }

    #[test]
    fn a_nul_byte_in_the_template_is_reported() {
        assert_eq!(
            LlamaChatTemplate::new("null\0byte"),
            Err(CString::new("null\0byte").unwrap_err())
        );
    }

    #[test]
    fn debug_formats_a_template_as_its_quoted_name() {
        let template = LlamaChatTemplate::new("chatml").unwrap();
        let debug_output = format!("{template:?}");

        assert_eq!(debug_output, "\"chatml\"");
    }

    #[test]
    fn to_string_copies_the_template_name() {
        let template = LlamaChatTemplate::new("llama3").unwrap();
        let owned = template.to_string().unwrap();

        assert_eq!(owned, "llama3");
    }

    #[test]
    fn rendering_substitutes_the_globals_into_the_template() {
        let template = LlamaChatTemplate::new(
            "{% for option in options %}[{{ option }}]{% endfor %} {{ question }}",
        )
        .unwrap();

        assert_eq!(
            template.render(r#"{"question": "Is it raining?", "options": ["yes", "no"]}"#),
            Ok("[yes][no] Is it raining?".to_owned())
        );
    }

    #[test]
    fn globals_with_a_nul_byte_are_refused_before_rendering() {
        let template = LlamaChatTemplate::new("{{ question }}").unwrap();
        let globals = "{\x00}";

        assert_eq!(
            template.render(globals),
            Err(ModelError::TemplateGlobalsContainNulByte(
                CString::new(globals).unwrap_err()
            ))
        );
    }

    #[test]
    fn globals_that_are_not_json_are_reported_with_the_parser_message() {
        let template = LlamaChatTemplate::new("{{ question }}").unwrap();

        assert_eq!(
            template.render("not valid json at all"),
            Err(ModelError::TemplateGlobalsInvalidJson {
                message: "[json.exception.parse_error.101] parse error at line 1, column 2: \
                          syntax error while parsing value - invalid literal; last read: 'no'"
                    .to_owned(),
            })
        );
    }

    #[test]
    fn globals_that_are_not_a_json_object_are_refused() {
        let template = LlamaChatTemplate::new("{{ question }}").unwrap();

        assert_eq!(
            template.render(r#"["question"]"#),
            Err(ModelError::TemplateGlobalsNotAJsonObject)
        );
    }

    #[test]
    fn a_template_with_a_syntax_error_is_reported_with_the_parser_message() {
        let template = LlamaChatTemplate::new("{% if question %}unterminated").unwrap();

        assert_eq!(
            template.render("{}"),
            Err(ModelError::TemplateInvalid {
                message: "Unexpected token type".to_owned(),
            })
        );
    }

    #[test]
    fn a_template_that_fails_while_executing_is_reported_with_the_runtime_message() {
        let template = LlamaChatTemplate::new("{{ raise_exception('no question') }}").unwrap();

        assert_eq!(
            template.render("{}"),
            Err(ModelError::TemplateExecutionFailed {
                message: "\n------------\nWhile executing CallExpression at line 1, column 19 in \
                          source:\n...{{ raise_exception('no question') }}...\n                     \
                          ^\nError: Jinja Exception: no question"
                    .to_owned(),
            })
        );
    }

    #[test]
    fn as_c_str_views_the_template_name() {
        let template = LlamaChatTemplate::new("test").unwrap();
        let cstr = template.as_c_str();

        assert_eq!(cstr.to_str().unwrap(), "test");
    }

    #[test]
    fn a_rendered_text_that_is_not_utf8_is_reported_with_its_bytes() {
        let not_utf8_text: [u8; 2] = [0xFF, 0];
        let out_rendered = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                not_utf8_text.as_ptr().cast::<c_char>(),
            )
        };

        assert_eq!(
            unsafe {
                render_template_status_to_result(
                    llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_OK,
                    out_rendered,
                    ptr::null_mut(),
                )
            },
            Err(ModelError::RenderedTemplateNotUtf8 {
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            })
        );
    }

    #[test]
    fn an_exception_llama_cpp_throws_is_reported_with_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"rendering threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                render_template_status_to_result(
                    llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                )
            },
            Err(ModelError::RenderReported {
                message: "rendering threw".to_owned(),
            })
        );
    }

    #[test]
    fn rendering_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
                llama_rs_render_template_status::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                render_template_status_to_result(status, ptr::null_mut(), ptr::null_mut())
            }),
            [
                Err(ModelError::RenderNotEnoughMemory),
                Err(ModelError::RenderLlamaCppOutOfMemory),
            ]
        );
    }
}
