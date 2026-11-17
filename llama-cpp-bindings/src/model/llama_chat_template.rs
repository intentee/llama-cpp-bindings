use std::ffi::NulError;
use std::ffi::{CStr, CString, c_char};
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ptr;
use std::str::Utf8Error;

use llama_cpp_ffi_status::cpp_string_error::CppStringError;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::error::chat_template_render_error::ChatTemplateRenderError;

/// # Safety
///
/// `out_rendered` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_render_template` call (or null). `out_rendered` is read and freed only on
/// `LLAMA_RS_RENDER_TEMPLATE_OK`, `out_error` only in the arms that carry a message.
unsafe fn render_template_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_render_template_status,
    out_rendered: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ChatTemplateRenderError> {
    let reported_message = || -> Result<String, CppStringError> {
        unsafe {
            read_and_free_cpp_string(
                out_error,
                "llama_rs_render_template",
                "reported a thrown C++ exception without an error message",
            )
        }
    };
    let contract_violation = |detail: &'static str| -> Result<String, ChatTemplateRenderError> {
        Err(FfiContractError {
            operation: "llama_rs_render_template",
            detail,
        }
        .into())
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_OK => Ok(unsafe {
            read_and_free_cpp_string(
                out_rendered,
                "llama_rs_render_template",
                "success status contained a null rendered string",
            )
        }?),
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ChatTemplateRenderError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ChatTemplateRenderError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_INVALID_JSON => {
            Err(ChatTemplateRenderError::InvalidJson {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_GLOBALS_NOT_A_JSON_OBJECT => {
            Err(ChatTemplateRenderError::GlobalsNotAJsonObject)
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_INVALID_TEMPLATE => {
            Err(ChatTemplateRenderError::InvalidTemplate {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_EXECUTION_FAILED => {
            Err(ChatTemplateRenderError::ExecutionFailed {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(ChatTemplateRenderError::Reported {
                message: reported_message()?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_TEMPLATE_SOURCE_ARG => {
            contract_violation("was given a null template_source argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_GLOBALS_JSON_ARG => {
            contract_violation("was given a null globals_json argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_OUT_RENDERED_ARG => {
            contract_violation("was given a null out_rendered argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_OUT_ERROR_ARG => {
            contract_violation("was given a null out_error argument")
        }
        other => Err(FfiStatusError {
            operation: "llama_rs_render_template",
            code: i64::from(other),
        }
        .into()),
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
    /// Returns [`ChatTemplateRenderError`] when `globals_json` contains a NUL byte, is not valid
    /// JSON or not a JSON object, the template cannot be parsed or executed, the wrapper fails,
    /// or the rendered text is not valid UTF-8.
    pub fn render(&self, globals_json: &str) -> Result<String, ChatTemplateRenderError> {
        let globals = CString::new(globals_json)?;
        let mut out_rendered: *mut c_char = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_render_template(
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

    use llama_cpp_ffi_status::cpp_string_error::CppStringError;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;

    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use crate::error::chat_template_render_error::ChatTemplateRenderError;

    use super::render_template_status_to_result;
    use crate::model::llama_chat_template::LlamaChatTemplate;

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
            Err(ChatTemplateRenderError::GlobalsContainNulByte(
                CString::new(globals).unwrap_err()
            ))
        );
    }

    #[test]
    fn globals_that_are_not_json_are_reported_with_the_parser_message() {
        let template = LlamaChatTemplate::new("{{ question }}").unwrap();

        assert_eq!(
            template.render("not valid json at all"),
            Err(ChatTemplateRenderError::InvalidJson {
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
            Err(ChatTemplateRenderError::GlobalsNotAJsonObject)
        );
    }

    #[test]
    fn a_template_with_a_syntax_error_is_reported_with_the_parser_message() {
        let template = LlamaChatTemplate::new("{% if question %}unterminated").unwrap();

        assert_eq!(
            template.render("{}"),
            Err(ChatTemplateRenderError::InvalidTemplate {
                message: "Unexpected token type".to_owned(),
            })
        );
    }

    #[test]
    fn a_template_that_fails_while_executing_is_reported_with_the_runtime_message() {
        let template = LlamaChatTemplate::new("{{ raise_exception('no question') }}").unwrap();

        assert_eq!(
            template.render("{}"),
            Err(ChatTemplateRenderError::ExecutionFailed {
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
            llama_cpp_bindings_sys::llama_rs_string_dup(not_utf8_text.as_ptr().cast::<c_char>())
        };

        assert_eq!(
            unsafe {
                render_template_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_OK,
                    out_rendered,
                    ptr::null_mut(),
                )
            },
            Err(ChatTemplateRenderError::CppString(
                CppStringError::NotUtf8 {
                    operation: "llama_rs_render_template",
                    source: String::from_utf8(vec![0xFF]).unwrap_err().utf8_error(),
                    bytes: vec![0xFF],
                }
            ))
        );
    }

    #[test]
    fn a_success_without_a_rendered_text_is_a_contract_violation() {
        assert_eq!(
            unsafe {
                render_template_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_OK,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            },
            Err(ChatTemplateRenderError::CppString(CppStringError::Missing(
                FfiContractError {
                    operation: "llama_rs_render_template",
                    detail: "success status contained a null rendered string",
                }
            )))
        );
    }

    #[test]
    fn each_status_that_carries_a_message_without_one_is_a_contract_violation() {
        for status in [
            llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_INVALID_JSON,
            llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_INVALID_TEMPLATE,
            llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_EXECUTION_FAILED,
            llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
        ] {
            assert_eq!(
                unsafe {
                    render_template_status_to_result(status, ptr::null_mut(), ptr::null_mut())
                },
                Err(ChatTemplateRenderError::CppString(CppStringError::Missing(
                    FfiContractError {
                        operation: "llama_rs_render_template",
                        detail: "reported a thrown C++ exception without an error message",
                    }
                )))
            );
        }
    }

    #[test]
    fn an_exception_llama_cpp_throws_is_reported_with_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"rendering threw".as_ptr()) };

        assert_eq!(
            unsafe {
                render_template_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                )
            },
            Err(ChatTemplateRenderError::Reported {
                message: "rendering threw".to_owned(),
            })
        );
    }

    #[test]
    fn rendering_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                render_template_status_to_result(status, ptr::null_mut(), ptr::null_mut())
            }),
            [
                Err(ChatTemplateRenderError::NotEnoughMemory),
                Err(ChatTemplateRenderError::LlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn rendering_reports_each_null_argument_as_a_contract_violation() {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_TEMPLATE_SOURCE_ARG,
                "was given a null template_source argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_GLOBALS_JSON_ARG,
                "was given a null globals_json argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_OUT_RENDERED_ARG,
                "was given a null out_rendered argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_RENDER_TEMPLATE_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                unsafe {
                    render_template_status_to_result(status, ptr::null_mut(), ptr::null_mut())
                },
                Err(ChatTemplateRenderError::FfiContract(FfiContractError {
                    operation: "llama_rs_render_template",
                    detail,
                }))
            );
        }
    }

    #[test]
    fn rendering_preserves_an_unknown_status() {
        assert_eq!(
            unsafe { render_template_status_to_result(255, ptr::null_mut(), ptr::null_mut()) },
            Err(ChatTemplateRenderError::FfiStatus(FfiStatusError {
                operation: "llama_rs_render_template",
                code: 255,
            }))
        );
    }
}
