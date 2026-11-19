use enumflags2::BitFlags;
use std::ffi::CString;
use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_init_from_file_status;
use llama_cpp_model::llama_model::LlamaModel;

use crate::mtmd_capability::MtmdCapability;
use crate::mtmd_context_params::MtmdContextParams;
use crate::mtmd_error::MtmdError;
use crate::reported_error::reported_error;

fn capability_flags(
    llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
        chat,
        vision,
        audio,
        video,
        audio_stream,
        text_gen,
        audio_gen,
        audio_gen_spk_ref,
    }: llama_cpp_bindings_sys::bindings::mtmd_helper_cap,
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

/// # Safety
///
/// On `LLAMA_RS_MTMD_INIT_FROM_FILE_OK` `out_ctx` must be the non-null context llama.cpp created,
/// and on `LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the
/// non-null message the C++ side allocated; the message is freed here.
unsafe fn map_init_from_file_status(
    status: llama_rs_mtmd_init_from_file_status,
    out_ctx: *mut llama_cpp_bindings_sys::bindings::mtmd_context,
    out_error: *mut c_char,
    mmproj_path: &str,
) -> Result<MtmdContext, MtmdError> {
    match status {
        llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_OK => Ok(MtmdContext {
            context: unsafe { NonNull::new_unchecked(out_ctx) },
        }),
        llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL => {
            Err(MtmdError::MmprojUnloadable {
                path: PathBuf::from(mmproj_path),
            })
        }
        llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::InitNotEnoughMemory)
        }
        llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::InitLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe {
                reported_error("llama_rs_mtmd_init_from_file", out_error, |message| {
                    MtmdError::InitReported { message }
                })
            })
        }
    }
}

#[derive(Debug)]
pub struct MtmdContext {
    pub context: NonNull<llama_cpp_bindings_sys::bindings::mtmd_context>,
}

unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// # Errors
    ///
    /// Returns an [`MtmdError`] when the path or the context params cannot be passed to llama.cpp,
    /// or when llama.cpp fails to load the projector.
    pub fn init_from_file(
        mmproj_path: &str,
        LlamaModel { model, .. }: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdError> {
        let path_cstr = CString::new(mmproj_path).map_err(MtmdError::MmprojPathContainsNul)?;
        let ctx_params = llama_cpp_bindings_sys::bindings::mtmd_context_params::try_from(params)
            .map_err(Box::new)
            .map_err(MtmdError::InitContextParams)?;

        let mut out_ctx: *mut llama_cpp_bindings_sys::bindings::mtmd_context = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_init_from_file(
                path_cstr.as_ptr(),
                model.as_ptr(),
                ctx_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };

        unsafe { map_init_from_file_status(status, out_ctx, out_error, mmproj_path) }
    }

    #[must_use]
    pub fn capabilities(&self, LlamaModel { model, .. }: &LlamaModel) -> BitFlags<MtmdCapability> {
        capability_flags(unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_helper_get_capabilities(
                model.as_ptr(),
                self.context.as_ptr(),
            )
        })
    }

    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let rate = unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_get_audio_sample_rate(self.context.as_ptr())
        };
        (rate > 0).then_some(rate.unsigned_abs())
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use enumflags2::BitFlags;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_init_from_file_status;

    use super::capability_flags;
    use super::map_init_from_file_status;
    use crate::mtmd_capability::MtmdCapability;
    use crate::mtmd_error::MtmdError;

    const NO_CAPABILITIES: llama_cpp_bindings_sys::bindings::mtmd_helper_cap =
        llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
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
    fn init_from_file_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = unsafe {
            map_init_from_file_status(
                llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
                "mmproj.gguf",
            )
        };

        assert_eq!(result.unwrap_err(), MtmdError::InitNotEnoughMemory);
    }

    #[test]
    fn map_init_from_file_status_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            map_init_from_file_status(
                llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
                "",
            )
        };
        assert_eq!(outcome.err(), Some(MtmdError::InitLlamaCppOutOfMemory));
    }

    #[test]
    fn map_init_from_file_status_maps_a_thrown_exception_with_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"projector threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                map_init_from_file_status(
                    llama_rs_mtmd_init_from_file_status::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                    "mmproj.gguf",
                )
            }
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
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                chat: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                vision: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                audio: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                video: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                audio_stream: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                text_gen: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
                audio_gen: true,
                ..NO_CAPABILITIES
            },
            llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
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
        let text_to_speech = llama_cpp_bindings_sys::bindings::mtmd_helper_cap {
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
