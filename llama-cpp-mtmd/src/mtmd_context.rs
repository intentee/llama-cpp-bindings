use enumflags2::BitFlags;
use std::ffi::CString;
use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_wrapper_string::read_and_free_exception_message::read_and_free_exception_message;

use crate::mtmd_capability::MtmdCapability;
use crate::mtmd_context_params::MtmdContextParams;
use crate::mtmd_error::MtmdError;

fn capability_flags(
    llama_cpp_bindings_sys::mtmd_helper_cap {
        chat,
        vision,
        audio,
        video,
        audio_stream,
        text_gen,
        audio_gen,
        audio_gen_spk_ref,
    }: llama_cpp_bindings_sys::mtmd_helper_cap,
) -> BitFlags<MtmdCapability> {
    [
        (chat, MtmdCapability::Chat),
        (vision, MtmdCapability::Vision),
        (audio, MtmdCapability::Audio),
        (video, MtmdCapability::Video),
        (audio_stream, MtmdCapability::AudioStream),
        (text_gen, MtmdCapability::TextGeneration),
        (audio_gen, MtmdCapability::AudioGeneration),
        (
            audio_gen_spk_ref,
            MtmdCapability::AudioGenerationSpeakerReference,
        ),
    ]
    .into_iter()
    .filter(|(is_capable, _capability)| *is_capable)
    .map(|(_is_capable, capability)| capability)
    .collect()
}

fn map_init_from_file_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_init_from_file_status,
    out_ctx: *mut llama_cpp_bindings_sys::mtmd_context,
    out_error: *mut c_char,
    mmproj_path: &str,
) -> Result<MtmdContext, MtmdError> {
    const OPERATION: &str = "llama_rs_mtmd_init_from_file";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_OK => {
            let context =
                NonNull::new(out_ctx).ok_or(MtmdError::WrapperSucceededWithoutOutput {
                    operation: OPERATION,
                    output: "multimodal context",
                })?;
            Ok(MtmdContext { context })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL => {
            Err(MtmdError::MmprojUnloadable {
                path: PathBuf::from(mmproj_path),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::InitNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::InitLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_exception_message(OPERATION, out_error) }?;
            Err(MtmdError::InitReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG => {
            Err(MtmdError::WrapperRefusedNullArgument {
                operation: OPERATION,
                argument: "mmproj_path",
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG => {
            Err(MtmdError::WrapperRefusedNullArgument {
                operation: OPERATION,
                argument: "text_model",
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG => {
            Err(MtmdError::WrapperRefusedNullArgument {
                operation: OPERATION,
                argument: "out_ctx",
            })
        }
        other => Err(MtmdError::UnknownWrapperStatus {
            operation: OPERATION,
            code: i64::from(other),
        }),
    }
}

#[derive(Debug)]
pub struct MtmdContext {
    pub context: NonNull<llama_cpp_bindings_sys::mtmd_context>,
}

unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// # Errors
    ///
    /// Returns an [`MtmdError`] variant matching the wrapper's status code.
    pub fn init_from_file(
        mmproj_path: &str,
        LlamaModel { model, .. }: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdError> {
        let path_cstr = CString::new(mmproj_path).map_err(MtmdError::MmprojPathContainsNul)?;
        let ctx_params = llama_cpp_bindings_sys::mtmd_context_params::try_from(params)
            .map_err(Box::new)
            .map_err(MtmdError::InitContextParams)?;

        let mut out_ctx: *mut llama_cpp_bindings_sys::mtmd_context = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_init_from_file(
                path_cstr.as_ptr(),
                model.as_ptr(),
                ctx_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };

        map_init_from_file_status(status, out_ctx, out_error, mmproj_path)
    }

    #[must_use]
    pub fn capabilities(&self, LlamaModel { model, .. }: &LlamaModel) -> BitFlags<MtmdCapability> {
        capability_flags(unsafe {
            llama_cpp_bindings_sys::mtmd_helper_get_capabilities(
                model.as_ptr(),
                self.context.as_ptr(),
            )
        })
    }

    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let rate =
            unsafe { llama_cpp_bindings_sys::mtmd_get_audio_sample_rate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use enumflags2::BitFlags;
    use std::ptr;

    use llama_cpp_wrapper_string::wrapper_string_error::WrapperStringError;

    use super::capability_flags;
    use super::map_init_from_file_status;
    use crate::mtmd_capability::MtmdCapability;
    use crate::mtmd_error::MtmdError;

    const NO_CAPABILITIES: llama_cpp_bindings_sys::mtmd_helper_cap =
        llama_cpp_bindings_sys::mtmd_helper_cap {
            chat: false,
            vision: false,
            audio: false,
            video: false,
            audio_stream: false,
            text_gen: false,
            audio_gen: false,
            audio_gen_spk_ref: false,
        };

    #[test]
    fn init_from_file_success_with_null_context_is_a_success_without_output() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_OK,
            ptr::null_mut(),
            ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            MtmdError::WrapperSucceededWithoutOutput {
                operation: "llama_rs_mtmd_init_from_file",
                output: "multimodal context"
            }
        );
    }

    #[test]
    fn init_from_file_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
            ptr::null_mut(),
            ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(result.unwrap_err(), MtmdError::InitNotEnoughMemory);
    }

    #[test]
    fn init_from_file_status_reports_an_exception_without_a_message_as_a_missing_exception_message()
    {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            ptr::null_mut(),
            ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            MtmdError::WrapperString(WrapperStringError::ExceptionMessageMissing {
                operation: "llama_rs_mtmd_init_from_file"
            })
        );
    }

    #[test]
    fn init_from_file_unknown_status_is_preserved() {
        let result =
            map_init_from_file_status(255, ptr::null_mut(), ptr::null_mut(), "mmproj.gguf");

        assert_eq!(
            result.unwrap_err(),
            MtmdError::UnknownWrapperStatus {
                operation: "llama_rs_mtmd_init_from_file",
                code: 255
            }
        );
    }

    #[test]
    fn map_init_from_file_status_reports_a_null_mmproj_path_arg_as_a_refused_null_argument() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome.err(),
            Some(MtmdError::WrapperRefusedNullArgument {
                operation: "llama_rs_mtmd_init_from_file",
                argument: "mmproj_path"
            })
        );
    }

    #[test]
    fn map_init_from_file_status_reports_a_null_text_model_arg_as_a_refused_null_argument() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome.err(),
            Some(MtmdError::WrapperRefusedNullArgument {
                operation: "llama_rs_mtmd_init_from_file",
                argument: "text_model"
            })
        );
    }

    #[test]
    fn map_init_from_file_status_reports_a_null_out_ctx_arg_as_a_refused_null_argument() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome.err(),
            Some(MtmdError::WrapperRefusedNullArgument {
                operation: "llama_rs_mtmd_init_from_file",
                argument: "out_ctx"
            })
        );
    }

    #[test]
    fn map_init_from_file_status_reports_llama_cpp_running_out_of_memory() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(outcome.err(), Some(MtmdError::InitLlamaCppOutOfMemory));
    }

    #[test]
    fn map_init_from_file_status_maps_a_thrown_exception_with_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"projector threw".as_ptr()) };

        assert_eq!(
            map_init_from_file_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
                "mmproj.gguf",
            )
            .err(),
            Some(MtmdError::InitReported {
                message: "projector threw".to_owned(),
            })
        );
    }

    #[test]
    fn a_model_without_capabilities_has_no_flags() {
        assert_eq!(capability_flags(NO_CAPABILITIES), BitFlags::empty());
    }

    #[test]
    fn every_capability_of_llama_cpp_maps_to_its_own_flag() {
        let single_capabilities = [
            llama_cpp_bindings_sys::mtmd_helper_cap {
                chat: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                vision: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                audio: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                video: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                audio_stream: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                text_gen: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                audio_gen: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::mtmd_helper_cap {
                audio_gen_spk_ref: true,
                ..NO_CAPABILITIES
            },
        ];

        assert_eq!(
            single_capabilities.map(capability_flags),
            [
                BitFlags::from_flag(MtmdCapability::Chat),
                BitFlags::from_flag(MtmdCapability::Vision),
                BitFlags::from_flag(MtmdCapability::Audio),
                BitFlags::from_flag(MtmdCapability::Video),
                BitFlags::from_flag(MtmdCapability::AudioStream),
                BitFlags::from_flag(MtmdCapability::TextGeneration),
                BitFlags::from_flag(MtmdCapability::AudioGeneration),
                BitFlags::from_flag(MtmdCapability::AudioGenerationSpeakerReference),
            ]
        );
    }

    #[test]
    fn a_text_to_speech_model_generates_audio_but_cannot_chat() {
        let text_to_speech = llama_cpp_bindings_sys::mtmd_helper_cap {
            audio_gen: true,
            audio_gen_spk_ref: true,
            ..NO_CAPABILITIES
        };

        assert_eq!(
            capability_flags(text_to_speech),
            MtmdCapability::AudioGeneration | MtmdCapability::AudioGenerationSpeakerReference
        );
    }
}
