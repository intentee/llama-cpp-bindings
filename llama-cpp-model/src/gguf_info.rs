use std::ffi::c_char;
use std::path::Path;
use std::ptr;

use llama_cpp_bindings_sys::bindings::llama_rs_gguf_info;
use llama_cpp_bindings_sys::bindings::llama_rs_read_gguf_info_status;
use llama_cpp_path_c_string::path_c_string::path_c_string;

use crate::decision_type::DecisionType;
use crate::model_error::ModelError;
use crate::read_exception_message::read_exception_message;

/// # Safety
///
/// On `LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null
/// message the C++ side allocated; it is freed here.
unsafe fn read_gguf_info_status_to_result(
    status: llama_rs_read_gguf_info_status,
    llama_rs_gguf_info {
        decision_type,
        n_ctx_train,
        has_chat_template,
    }: llama_rs_gguf_info,
    out_error: *mut c_char,
) -> Result<GgufInfo, ModelError> {
    match status {
        llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_OK => Ok(GgufInfo {
            decision_type: DecisionType::try_from(decision_type)?,
            n_ctx_train,
            has_chat_template,
        }),
        llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::GgufInfoNotEnoughMemory)
        }
        llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::GgufInfoLlamaCppOutOfMemory)
        }
        llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_exception_message("llama_rs_read_gguf_info", out_error) }?;
            Err(ModelError::GgufInfoReported { message })
        }
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
    /// Returns [`ModelError`] when `path` is not UTF-8 or contains a NUL byte, llama.cpp
    /// fails to read the file's metadata, or the decision type read is not one
    /// `llama_rs_decision_type` defines.
    pub fn read(path: &Path) -> Result<Self, ModelError> {
        let path_c_string = path_c_string(path).map_err(ModelError::GgufInfoPathInvalid)?;
        let mut out_info = llama_rs_gguf_info {
            decision_type: llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_UNKNOWN,
            n_ctx_train: 0,
            has_chat_template: false,
        };
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_read_gguf_info(
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
    use std::ffi::OsStr;
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::process;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_gguf_info;
    use llama_cpp_bindings_sys::bindings::llama_rs_read_gguf_info_status;
    use llama_cpp_gguf::gguf_context::GgufContext;
    use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

    use super::GgufInfo;
    use super::read_gguf_info_status_to_result;
    use crate::decision_type::DecisionType;
    use crate::model_error::ModelError;

    const VOCABULARY_FIXTURE_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../llama-cpp-gguf/fixtures/ggml-vocab-bert-bge.gguf"
    );

    const NOT_A_DECISION_MODEL: llama_rs_gguf_info = llama_rs_gguf_info {
        decision_type: llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_NONE,
        n_ctx_train: 0,
        has_chat_template: false,
    };

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
        let mut context = GgufContext::from_file(VOCABULARY_FIXTURE_PATH).unwrap();
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
            Err(ModelError::GgufInfoPathInvalid(
                PathCStringError::ContainsNulByte(CString::new(path).unwrap_err())
            ))
        );
    }

    #[test]
    fn a_path_that_is_not_utf8_is_refused_before_the_file_is_read() {
        let path = Path::new(OsStr::from_bytes(b"/models/\xff\xfe.gguf"));

        assert_eq!(
            GgufInfo::read(path),
            Err(ModelError::GgufInfoPathInvalid(PathCStringError::NotUtf8(
                path.to_path_buf()
            )))
        );
    }

    #[test]
    fn a_decision_type_llama_rs_decision_type_does_not_define_is_refused() {
        let result = unsafe {
            read_gguf_info_status_to_result(
                llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_OK,
                llama_rs_gguf_info {
                    decision_type: 9999,
                    ..NOT_A_DECISION_MODEL
                },
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ModelError::UnknownDecisionType(9999)));
    }

    #[test]
    fn reading_gguf_info_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_ERROR_STRING_ALLOCATION_FAILED,
                llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                read_gguf_info_status_to_result(status, NOT_A_DECISION_MODEL, ptr::null_mut())
            }),
            [
                Err(ModelError::GgufInfoNotEnoughMemory),
                Err(ModelError::GgufInfoLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn reading_gguf_info_reports_an_exception_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"metadata read threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                read_gguf_info_status_to_result(
                    llama_rs_read_gguf_info_status::LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    NOT_A_DECISION_MODEL,
                    out_error,
                )
            },
            Err(ModelError::GgufInfoReported {
                message: "metadata read threw".to_owned()
            })
        );
    }
}
