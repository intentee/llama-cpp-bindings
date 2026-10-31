use std::ffi::CStr;
use std::ffi::c_char;
use std::num::NonZeroU32;
use std::ptr;

use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::backend_device_index::backend_device_index;

use super::mtmd_context_params::MtmdContextParams;
use super::mtmd_image_token_limit::MtmdImageTokenLimit;
use super::mtmd_memory_usage_error::MtmdMemoryUsageError;

const OPERATION: &str = "llama_rs_mtmd_get_memory_usage";

fn get_memory_usage_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_get_memory_usage_status,
    out_error: *mut c_char,
) -> Result<(), MtmdMemoryUsageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_MMPROJ_PATH_ARG => {
            Err(contract_violation("was given a null mmproj_path argument"))
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_DEVICE_MEMORY_ARG => Err(
            contract_violation("was given a null out_device_memory argument"),
        ),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_DEVICE_MEMORY_COUNT_ARG => {
            Err(contract_violation(
                "was given a null out_device_memory_count argument",
            ))
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_IMAGE_MAX_TOKENS_ARG => {
            Err(contract_violation(
                "was given a null out_image_max_tokens argument",
            ))
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_USE_NON_CAUSAL_ARG => Err(
            contract_violation("was given a null out_use_non_causal argument"),
        ),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_DEVICE_MEMORY_CAPACITY_EXCEEDED => {
            Err(contract_violation(
                "measured memory on more devices than the backend registers",
            ))
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdMemoryUsageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdMemoryUsageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    OPERATION,
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(MtmdMemoryUsageError::Reported { message })
        }
        other => Err(llama_cpp_ffi_status::FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

const fn contract_violation(detail: &'static str) -> MtmdMemoryUsageError {
    MtmdMemoryUsageError::FfiContract(llama_cpp_ffi_status::FfiContractError {
        operation: OPERATION,
        detail,
    })
}

/// The worst-case memory a multimodal projector takes on each backend device, with the image
/// budget it was measured at, as llama.cpp estimates it before the projector is loaded.
#[derive(Debug)]
pub struct MtmdMemoryUsage {
    device_memory: Vec<llama_cpp_bindings_sys::llama_rs_mtmd_device_memory>,
    image_max_tokens: i32,
    decodes_images_non_causally: bool,
}

impl MtmdMemoryUsage {
    /// Measures the memory the projector at `mmproj_path` takes when loaded with `params`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdMemoryUsageError::Unloadable`] when llama.cpp cannot load the projector,
    /// [`MtmdMemoryUsageError::ContextParams`] when `params` have no C representation, or the
    /// variant matching the wrapper's status code.
    pub fn measure(
        mmproj_path: &CStr,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdMemoryUsageError> {
        let context_params = llama_cpp_bindings_sys::mtmd_context_params::try_from(params)?;
        let device_count = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() };
        let mut device_memory = Vec::with_capacity(device_count);
        let mut device_memory_count: usize = 0;
        let mut image_max_tokens: i32 = 0;
        let mut decodes_images_non_causally = false;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_get_memory_usage(
                mmproj_path.as_ptr(),
                context_params,
                device_memory.as_mut_ptr(),
                device_count,
                &raw mut device_memory_count,
                &raw mut image_max_tokens,
                &raw mut decodes_images_non_causally,
                &raw mut out_error,
            )
        };

        get_memory_usage_status_to_result(status, out_error)?;

        unsafe { device_memory.set_len(device_memory_count) };

        if device_memory.is_empty() {
            return Err(MtmdMemoryUsageError::Unloadable {
                path: mmproj_path.to_owned(),
            });
        }

        Ok(Self {
            device_memory,
            image_max_tokens,
            decodes_images_non_causally,
        })
    }

    /// Adds the projector's memory to the fit margin of each device it takes memory on, so
    /// fitting the text model leaves that memory free, as llama-server does. `margins` is
    /// indexed by backend device, like the margins of
    /// [`LlamaModelParams::fit_params`](crate::model::params::LlamaModelParams::fit_params);
    /// a device past its end has no margin to grow.
    pub fn add_to_fit_margins(&self, margins: &mut [usize]) {
        for llama_cpp_bindings_sys::llama_rs_mtmd_device_memory { device, bytes } in
            &self.device_memory
        {
            let margin = backend_device_index(*device)
                .and_then(|device_index| margins.get_mut(device_index));

            if let Some(margin) = margin {
                *margin += bytes;
            }
        }
    }

    /// Lowers the image token limits of `params` to `micro_batch_tokens` when the projector
    /// decodes images non-causally and could produce more tokens than one micro-batch holds,
    /// because a non-causal image must be decoded in a single micro-batch.
    #[must_use]
    pub fn fit_image_tokens_into_micro_batch(
        &self,
        params: MtmdContextParams,
        micro_batch_tokens: NonZeroU32,
    ) -> MtmdContextParams {
        if !self.decodes_images_non_causally
            || i64::from(self.image_max_tokens) <= i64::from(micro_batch_tokens.get())
        {
            return params;
        }

        MtmdContextParams {
            image_min_tokens: params.image_min_tokens.at_most(micro_batch_tokens),
            image_max_tokens: MtmdImageTokenLimit::Tokens(micro_batch_tokens),
            ..params
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::ptr;

    use llama_cpp_fault_injection::ArmedFault;
    use llama_cpp_fault_injection::injected_fault_message;

    use super::get_memory_usage_status_to_result;
    use crate::mtmd::mtmd_context_params::MtmdContextParams;
    use crate::mtmd::mtmd_context_params_error::MtmdContextParamsError;
    use crate::mtmd::mtmd_image_token_limit::MtmdImageTokenLimit;
    use crate::mtmd::mtmd_memory_usage::MtmdMemoryUsage;
    use crate::mtmd::mtmd_memory_usage_error::MtmdMemoryUsageError;

    const MICRO_BATCH_TOKENS: NonZeroU32 = NonZeroU32::new(64).unwrap();

    fn first_device_usage(bytes: usize) -> MtmdMemoryUsage {
        MtmdMemoryUsage {
            device_memory: vec![llama_cpp_bindings_sys::llama_rs_mtmd_device_memory {
                device: unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(0) },
                bytes,
            }],
            image_max_tokens: -1,
            decodes_images_non_causally: false,
        }
    }

    fn image_usage(image_max_tokens: i32, decodes_images_non_causally: bool) -> MtmdMemoryUsage {
        MtmdMemoryUsage {
            device_memory: Vec::new(),
            image_max_tokens,
            decodes_images_non_causally,
        }
    }

    fn params_with_min_tokens(image_min_tokens: MtmdImageTokenLimit) -> MtmdContextParams {
        MtmdContextParams {
            image_min_tokens,
            ..MtmdContextParams::default()
        }
    }

    fn contract_violation(detail: &'static str) -> MtmdMemoryUsageError {
        llama_cpp_ffi_status::FfiContractError {
            operation: "llama_rs_mtmd_get_memory_usage",
            detail,
        }
        .into()
    }

    #[test]
    fn measuring_a_missing_projector_reports_it_unloadable() {
        assert_eq!(
            MtmdMemoryUsage::measure(c"/nonexistent/mmproj.gguf", &MtmdContextParams::default())
                .err(),
            Some(MtmdMemoryUsageError::Unloadable {
                path: c"/nonexistent/mmproj.gguf".to_owned(),
            })
        );
    }

    #[test]
    fn a_measurement_llama_cpp_fails_is_reported() {
        let _fault = ArmedFault::at(
            llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_GET_MEMORY_USAGE_THROWS,
        );

        assert_eq!(
            MtmdMemoryUsage::measure(c"/nonexistent/mmproj.gguf", &MtmdContextParams::default())
                .err(),
            Some(MtmdMemoryUsageError::Reported {
                message: injected_fault_message(),
            })
        );
    }

    #[test]
    fn measuring_with_unrepresentable_params_is_refused() {
        let params = MtmdContextParams {
            image_max_tokens: MtmdImageTokenLimit::Tokens(NonZeroU32::MAX),
            ..MtmdContextParams::default()
        };

        assert_eq!(
            MtmdMemoryUsage::measure(c"/nonexistent/mmproj.gguf", &params).err(),
            i32::try_from(u32::MAX).err().map(|source| {
                MtmdMemoryUsageError::ContextParams(
                    MtmdContextParamsError::ImageMaxTokensExceedCInt {
                        limit: MtmdImageTokenLimit::Tokens(NonZeroU32::MAX),
                        source,
                    },
                )
            })
        );
    }

    #[test]
    fn projector_memory_grows_the_margin_of_its_device() {
        let mut margins = [1, 2];

        first_device_usage(5).add_to_fit_margins(&mut margins);

        assert_eq!(margins, [6, 2]);
    }

    #[test]
    fn projector_memory_on_a_device_without_a_margin_is_not_added() {
        let mut margins: [usize; 0] = [];

        first_device_usage(5).add_to_fit_margins(&mut margins);

        assert_eq!(margins, [0_usize; 0]);
    }

    #[test]
    fn a_causal_projector_keeps_the_image_token_limits() {
        let params = params_with_min_tokens(MtmdImageTokenLimit::Tokens(NonZeroU32::MAX));

        assert_eq!(
            image_usage(4096, false)
                .fit_image_tokens_into_micro_batch(params.clone(), MICRO_BATCH_TOKENS),
            params
        );
    }

    #[test]
    fn a_non_causal_projector_within_the_micro_batch_keeps_the_image_token_limits() {
        let params = params_with_min_tokens(MtmdImageTokenLimit::Tokens(NonZeroU32::MAX));

        assert_eq!(
            image_usage(64, true)
                .fit_image_tokens_into_micro_batch(params.clone(), MICRO_BATCH_TOKENS),
            params
        );
    }

    #[test]
    fn a_non_causal_projector_beyond_the_micro_batch_lowers_the_image_token_limits_to_it() {
        let params = params_with_min_tokens(MtmdImageTokenLimit::Tokens(NonZeroU32::MAX));

        assert_eq!(
            image_usage(65, true)
                .fit_image_tokens_into_micro_batch(params.clone(), MICRO_BATCH_TOKENS),
            MtmdContextParams {
                image_min_tokens: MtmdImageTokenLimit::Tokens(MICRO_BATCH_TOKENS),
                image_max_tokens: MtmdImageTokenLimit::Tokens(MICRO_BATCH_TOKENS),
                ..params
            }
        );
    }

    #[test]
    fn the_ok_status_is_success() {
        assert_eq!(
            get_memory_usage_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_OK,
                ptr::null_mut(),
            ),
            Ok(())
        );
    }

    #[test]
    fn wrapper_contract_statuses_are_contract_violations() {
        let statuses = [
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_MMPROJ_PATH_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_DEVICE_MEMORY_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_DEVICE_MEMORY_COUNT_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_IMAGE_MAX_TOKENS_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_USE_NON_CAUSAL_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_DEVICE_MEMORY_CAPACITY_EXCEEDED,
        ];

        assert_eq!(
            statuses.map(|status| get_memory_usage_status_to_result(status, ptr::null_mut())),
            [
                Err(contract_violation("was given a null mmproj_path argument")),
                Err(contract_violation(
                    "was given a null out_device_memory argument"
                )),
                Err(contract_violation(
                    "was given a null out_device_memory_count argument"
                )),
                Err(contract_violation(
                    "was given a null out_image_max_tokens argument"
                )),
                Err(contract_violation(
                    "was given a null out_use_non_causal argument"
                )),
                Err(contract_violation(
                    "measured memory on more devices than the backend registers"
                )),
            ]
        );
    }

    #[test]
    fn a_failed_error_string_allocation_is_not_enough_memory() {
        assert_eq!(
            get_memory_usage_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(MtmdMemoryUsageError::NotEnoughMemory)
        );
    }

    #[test]
    fn llama_cpp_running_out_of_memory_is_reported() {
        assert_eq!(
            get_memory_usage_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
            ),
            Err(MtmdMemoryUsageError::LlamaCppOutOfMemory)
        );
    }

    #[test]
    fn a_thrown_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"projector threw".as_ptr()) };

        assert_eq!(
            get_memory_usage_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                out_error,
            ),
            Err(MtmdMemoryUsageError::Reported {
                message: "projector threw".to_owned(),
            })
        );
    }

    #[test]
    fn a_thrown_exception_without_a_message_is_a_contract_violation() {
        assert_eq!(
            get_memory_usage_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(contract_violation(
                "reported a thrown C++ exception without an error message"
            ))
        );
    }

    #[test]
    fn an_unknown_status_is_preserved() {
        assert_eq!(
            get_memory_usage_status_to_result(255, ptr::null_mut()),
            Err(MtmdMemoryUsageError::FfiStatus(
                llama_cpp_ffi_status::FfiStatusError {
                    operation: "llama_rs_mtmd_get_memory_usage",
                    code: 255,
                }
            ))
        );
    }
}
