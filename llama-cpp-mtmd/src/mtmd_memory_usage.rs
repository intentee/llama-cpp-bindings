use std::ffi::CStr;
use std::ffi::c_char;
use std::num::NonZeroU32;
use std::ptr;

use llama_cpp_backend::backend_device_index::backend_device_index;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_get_memory_usage_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::mtmd_context_params::MtmdContextParams;
use crate::mtmd_error::MtmdError;
use crate::mtmd_image_token_limit::MtmdImageTokenLimit;

/// # Safety
///
/// On `LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the
/// non-null message the C++ side allocated; it is freed here.
unsafe fn get_memory_usage_status_to_result(
    status: llama_rs_mtmd_get_memory_usage_status,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    match status {
        llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_OK => Ok(()),
        llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_DEVICE_MEMORY_CAPACITY_EXCEEDED => {
            Err(MtmdError::MemoryUsageCoversMoreDevicesThanRegistered {
                operation: "llama_rs_mtmd_get_memory_usage",
            })
        }
        llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::MemoryUsageNotEnoughMemory)
        }
        llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::MemoryUsageLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_get_memory_usage", |message| {
                    MtmdError::MemoryUsageReported { message }
                }))
        }
    }
}

/// The worst-case memory a multimodal projector takes on each backend device, with the image
/// budget it was measured at, as llama.cpp estimates it before the projector is loaded.
#[derive(Debug)]
pub struct MtmdMemoryUsage {
    device_memory: Vec<llama_cpp_bindings_sys::bindings::llama_rs_mtmd_device_memory>,
    image_max_tokens: i32,
    decodes_images_non_causally: bool,
}

impl MtmdMemoryUsage {
    /// Measures the memory the projector at `mmproj_path` takes when loaded with `params`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::MemoryUsageUnloadable`] when llama.cpp cannot load the projector,
    /// [`MtmdError::MemoryUsageContextParams`] when `params` have no C representation, or the
    /// variant for llama.cpp failing to measure the projector.
    pub fn measure(mmproj_path: &CStr, params: &MtmdContextParams) -> Result<Self, MtmdError> {
        let context_params =
            llama_cpp_bindings_sys::bindings::mtmd_context_params::try_from(params)
                .map_err(Box::new)
                .map_err(MtmdError::MemoryUsageContextParams)?;
        let device_count = unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_dev_count() };
        let mut device_memory = Vec::with_capacity(device_count);
        let mut device_memory_count: usize = 0;
        let mut image_max_tokens: i32 = 0;
        let mut decodes_images_non_causally = false;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_get_memory_usage(
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

        unsafe { get_memory_usage_status_to_result(status, out_error) }?;

        unsafe { device_memory.set_len(device_memory_count) };

        if device_memory.is_empty() {
            return Err(MtmdError::MemoryUsageUnloadable {
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
    /// `LlamaModelParams::fit_params`;
    /// a device past its end has no margin to grow.
    pub fn add_to_fit_margins(&self, margins: &mut [usize]) {
        for llama_cpp_bindings_sys::bindings::llama_rs_mtmd_device_memory { device, bytes } in
            &self.device_memory
        {
            let margin = backend_device_index(*device)
                .and_then(|device_index| margins.get_mut(device_index));

            if let Some(margin) = margin {
                *margin = margin.saturating_add(*bytes);
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

    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_get_memory_usage_status;
    use llama_cpp_fault_injection::armed_fault::ArmedFault;
    use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;

    use super::get_memory_usage_status_to_result;
    use crate::mtmd_context_params::MtmdContextParams;
    use crate::mtmd_error::MtmdError;
    use crate::mtmd_image_token_limit::MtmdImageTokenLimit;
    use crate::mtmd_memory_usage::MtmdMemoryUsage;

    const MICRO_BATCH_TOKENS: NonZeroU32 = NonZeroU32::new(64).unwrap();

    fn first_device_usage(bytes: usize) -> MtmdMemoryUsage {
        MtmdMemoryUsage {
            device_memory: vec![
                llama_cpp_bindings_sys::bindings::llama_rs_mtmd_device_memory {
                    device: unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_dev_get(0) },
                    bytes,
                },
            ],
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

    #[test]
    fn measuring_a_missing_projector_reports_it_unloadable() {
        assert_eq!(
            MtmdMemoryUsage::measure(c"/nonexistent/mmproj.gguf", &MtmdContextParams::default())
                .err(),
            Some(MtmdError::MemoryUsageUnloadable {
                path: c"/nonexistent/mmproj.gguf".to_owned(),
            })
        );
    }

    #[test]
    fn a_measurement_llama_cpp_fails_is_reported() {
        let _fault = ArmedFault::at(
            llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_GET_MEMORY_USAGE_THROWS,
        );

        assert_eq!(
            MtmdMemoryUsage::measure(c"/nonexistent/mmproj.gguf", &MtmdContextParams::default())
                .err(),
            Some(MtmdError::MemoryUsageReported {
                message: injected_fault_message().unwrap(),
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
                MtmdError::MemoryUsageContextParams(Box::new(MtmdError::ImageMaxTokensExceedCInt {
                    limit: MtmdImageTokenLimit::Tokens(NonZeroU32::MAX),
                    source,
                }))
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
            unsafe {
                get_memory_usage_status_to_result(
                    llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_OK,
                    ptr::null_mut(),
                )
            },
            Ok(())
        );
    }

    #[test]
    fn more_measured_devices_than_registered_is_reported() {
        assert_eq!(
            unsafe {
                get_memory_usage_status_to_result(
                    llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_DEVICE_MEMORY_CAPACITY_EXCEEDED,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::MemoryUsageCoversMoreDevicesThanRegistered {
                operation: "llama_rs_mtmd_get_memory_usage"
            })
        );
    }

    #[test]
    fn a_failed_error_string_allocation_is_not_enough_memory() {
        assert_eq!(
            unsafe {
                get_memory_usage_status_to_result(
                    llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::MemoryUsageNotEnoughMemory)
        );
    }

    #[test]
    fn llama_cpp_running_out_of_memory_is_reported() {
        assert_eq!(
            unsafe {
                get_memory_usage_status_to_result(
                    llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::MemoryUsageLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn a_thrown_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"projector threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                get_memory_usage_status_to_result(
                    llama_rs_mtmd_get_memory_usage_status::LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    out_error,
                )
            },
            Err(MtmdError::MemoryUsageReported {
                message: "projector threw".to_owned(),
            })
        );
    }
}
