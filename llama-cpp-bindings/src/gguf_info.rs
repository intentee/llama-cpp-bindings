use std::ffi::c_char;
use std::path::Path;
use std::ptr;

use llama_cpp_bindings_sys::llama_rs_gguf_info;
use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::error::gguf_info_error::GgufInfoError;
use crate::model::decision_type::DecisionType;
use crate::path_c_string::path_c_string;

/// # Safety
///
/// `out_error` must be the pointer populated by the preceding `llama_rs_read_gguf_info` call (or
/// null); it is read and freed only in the CXX-exception arm.
unsafe fn read_gguf_info_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_read_gguf_info_status,
    llama_rs_gguf_info {
        decision_type,
        n_ctx_train,
        has_chat_template,
    }: llama_rs_gguf_info,
    out_error: *mut c_char,
) -> Result<GgufInfo, GgufInfoError> {
    let contract_violation = |detail: &'static str| -> Result<GgufInfo, GgufInfoError> {
        Err(FfiContractError {
            operation: "llama_rs_read_gguf_info",
            detail,
        }
        .into())
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_OK => Ok(GgufInfo {
            decision_type: DecisionType::try_from(decision_type)?,
            n_ctx_train,
            has_chat_template,
        }),
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GgufInfoError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GgufInfoError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_read_gguf_info",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(GgufInfoError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_NULL_PATH_ARG => {
            contract_violation("was given a null path argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_NULL_OUT_INFO_ARG => {
            contract_violation("was given a null out_info argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_NULL_OUT_ERROR_ARG => {
            contract_violation("was given a null out_error argument")
        }
        other => Err(FfiStatusError {
            operation: "llama_rs_read_gguf_info",
            code: i64::from(other),
        }
        .into()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufInfo {
    pub decision_type: DecisionType,
    pub n_ctx_train: u32,
    pub has_chat_template: bool,
}

impl GgufInfo {
    /// # Errors
    ///
    /// Returns [`GgufInfoError`] when `path` is not UTF-8 or contains a NUL byte, the wrapper
    /// fails, or it reports a decision type it does not define.
    pub fn read(path: &Path) -> Result<Self, GgufInfoError> {
        let path_c_string = path_c_string(path)?;
        let mut out_info = llama_rs_gguf_info {
            decision_type: llama_cpp_bindings_sys::LLAMA_RS_DECISION_TYPE_UNKNOWN,
            n_ctx_train: 0,
            has_chat_template: false,
        };
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_read_gguf_info(
                path_c_string.as_ptr(),
                &raw mut out_info,
                &raw mut out_error,
            )
        };

        unsafe { read_gguf_info_status_to_result(status, out_info, out_error) }
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::ffi::CString;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process;

    use super::GgufInfo;
    use crate::error::gguf_info_error::GgufInfoError;
    use crate::error::path_c_string_error::PathCStringError;
    use crate::gguf_context::GgufContext;
    use crate::model::decision_type::DecisionType;

    fn vocabulary_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("ggml-vocab-bert-bge.gguf")
    }

    #[test]
    fn a_missing_file_reads_as_an_unknown_decision_type_without_context_or_template() {
        assert_eq!(
            GgufInfo::read(Path::new("/nonexistent/decision.gguf")),
            Ok(GgufInfo {
                decision_type: DecisionType::Unknown,
                n_ctx_train: 0,
                has_chat_template: false,
            })
        );
    }

    #[test]
    fn the_decision_type_context_length_and_chat_template_of_a_file_are_read() {
        let decision_gguf_path = env::temp_dir().join(format!(
            "llama_cpp_bindings_gguf_info_{}_decision.gguf",
            process::id(),
        ));
        let mut context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        context.set_val_str(c"bert.decision.type", c"clef");
        context.set_val_u32(c"bert.context_length", 384);
        context.set_val_str(c"tokenizer.chat_template", c"{{ question }}");
        context.write_to_file(&decision_gguf_path).unwrap();

        let info = GgufInfo::read(&decision_gguf_path);
        fs::remove_file(&decision_gguf_path).unwrap();

        assert_eq!(
            info,
            Ok(GgufInfo {
                decision_type: DecisionType::Clef,
                n_ctx_train: 384,
                has_chat_template: true,
            })
        );
    }

    #[test]
    fn a_path_with_a_nul_byte_is_refused_before_the_file_is_read() {
        let path = "/models/deci\0sion.gguf";

        assert_eq!(
            GgufInfo::read(Path::new(path)),
            Err(GgufInfoError::Path(PathCStringError::ContainsNulByte(
                CString::new(path).unwrap_err()
            )))
        );
    }
}

#[cfg(test)]
mod read_gguf_info_status_to_result_tests {
    use std::ptr;

    use llama_cpp_bindings_sys::llama_rs_gguf_info;
    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::read_gguf_info_status_to_result;
    use crate::error::gguf_info_error::GgufInfoError;
    use crate::error::unknown_decision_type::UnknownDecisionType;

    const NOT_A_DECISION_MODEL: llama_rs_gguf_info = llama_rs_gguf_info {
        decision_type: llama_cpp_bindings_sys::LLAMA_RS_DECISION_TYPE_NONE,
        n_ctx_train: 0,
        has_chat_template: false,
    };

    #[test]
    fn a_decision_type_the_wrapper_does_not_define_is_refused() {
        let result = unsafe {
            read_gguf_info_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_OK,
                llama_rs_gguf_info {
                    decision_type: 9999,
                    ..NOT_A_DECISION_MODEL
                },
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(GgufInfoError::UnknownDecisionType(UnknownDecisionType(
                9999
            )))
        );
    }

    #[test]
    fn reading_gguf_info_reports_each_null_argument_as_a_contract_violation() {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_NULL_PATH_ARG,
                "was given a null path argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_NULL_OUT_INFO_ARG,
                "was given a null out_info argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                unsafe {
                    read_gguf_info_status_to_result(status, NOT_A_DECISION_MODEL, ptr::null_mut())
                },
                Err(GgufInfoError::FfiContract(FfiContractError {
                    operation: "llama_rs_read_gguf_info",
                    detail,
                }))
            );
        }
    }

    #[test]
    fn reading_gguf_info_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                read_gguf_info_status_to_result(status, NOT_A_DECISION_MODEL, ptr::null_mut())
            }),
            [
                Err(GgufInfoError::NotEnoughMemory),
                Err(GgufInfoError::LlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn reading_gguf_info_reports_an_exception_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"metadata read threw".as_ptr()) };

        assert_eq!(
            unsafe {
                read_gguf_info_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    NOT_A_DECISION_MODEL,
                    out_error,
                )
            },
            Err(GgufInfoError::Reported {
                message: "metadata read threw".to_owned()
            })
        );
    }

    #[test]
    fn reading_gguf_info_reports_an_exception_without_a_message_as_a_contract_violation() {
        assert_eq!(
            unsafe {
                read_gguf_info_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    NOT_A_DECISION_MODEL,
                    ptr::null_mut(),
                )
            },
            Err(GgufInfoError::CppString(CppStringError::Missing(
                FfiContractError {
                    operation: "llama_rs_read_gguf_info",
                    detail: "reported a thrown C++ exception without an error message",
                }
            )))
        );
    }

    #[test]
    fn reading_gguf_info_preserves_an_unknown_status() {
        assert_eq!(
            unsafe { read_gguf_info_status_to_result(255, NOT_A_DECISION_MODEL, ptr::null_mut()) },
            Err(GgufInfoError::FfiStatus(FfiStatusError {
                operation: "llama_rs_read_gguf_info",
                code: 255,
            }))
        );
    }
}
