use std::ffi::{CStr, CString, c_char};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::ptr;
use std::ptr::null;

use llama_cpp_backend::backend_device_index::backend_device_index;
use llama_cpp_backend::max_devices::max_devices;
use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::constants::EMPTY_KV_OVERRIDE;
use crate::fit_extra_model::FitExtraModel;
use crate::fit_result::FitResult;
use crate::kv_overrides::KvOverrides;
use crate::llama_context_params::LlamaContextParams;
use crate::llama_lazy_mode::LlamaLazyMode;
use crate::llama_load_mode::LlamaLoadMode;
use crate::llama_split_mode::LlamaSplitMode;
use crate::param_override_value::ParamOverrideValue;
use crate::params_error::ParamsError;

fn fit_params_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_fit_params_status,
    out_unrecognized_status_code: i32,
    out_error: *mut c_char,
) -> Result<(), ParamsError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_FAILURE => {
            Err(ParamsError::NoFittingMemoryLayout)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_ERROR => {
            Err(ParamsError::FitAborted)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_RETURNED_UNRECOGNIZED_STATUS_CODE => {
            Err(ParamsError::UnknownFitStatus {
                code: out_unrecognized_status_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParamsError::FitNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParamsError::FitLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_fit_params",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(ParamsError::FitReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_PATH_MODEL_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given a null path_model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_MPARAMS_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given a null mparams argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_CPARAMS_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given a null cparams argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_OUT_UNRECOGNIZED_STATUS_CODE_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given a null out_unrecognized_status_code argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_OUT_ERROR_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_EXTRA_PATH_MODEL_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given an extra model with a null path_model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_EXTRA_MPARAMS_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given an extra model with a null mparams argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_EXTRA_CPARAMS_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "was given an extra model with a null cparams argument",
            }
            .into())
        }
        other => Err(FfiError::UnknownStatus {
            operation: "llama_rs_fit_params",
            code: i64::from(other),
        }
        .into()),
    }
}

pub struct LlamaModelParams {
    pub params: llama_cpp_bindings_sys::llama_model_params,
    kv_overrides: Vec<llama_cpp_bindings_sys::llama_model_kv_override>,
    buft_overrides: Vec<llama_cpp_bindings_sys::llama_model_tensor_buft_override>,
    buft_override_patterns: Vec<CString>,
    devices: Pin<Box<[llama_cpp_bindings_sys::ggml_backend_dev_t]>>,
    tensor_split: Vec<f32>,
}

impl Debug for LlamaModelParams {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlamaModelParams")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("lazy_mode", &self.lazy_mode())
            .field("load_mode", &self.load_mode())
            .field("load_mtp", &self.params.load_mtp)
            .field("split_mode", &self.split_mode())
            .field("devices", &self.devices())
            .field("kv_overrides", &"vec of kv_overrides")
            .finish_non_exhaustive()
    }
}

impl LlamaModelParams {
    /// # Errors
    /// Returns [`ParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    ///
    pub fn append_kv_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
        value: ParamOverrideValue,
    ) -> Result<(), ParamsError> {
        let kv_override = self
            .kv_overrides
            .get_mut(0)
            .ok_or(ParamsError::NoAvailableOverrideSlot)?;

        if kv_override.key[0] != 0 {
            return Err(ParamsError::OverrideSlotNotEmpty);
        }

        let key_with_nul = key.to_bytes_with_nul();

        if key_with_nul.len() > kv_override.key.len() {
            return Err(ParamsError::OverrideKeyTooLong {
                length_with_nul: key_with_nul.len(),
                capacity: kv_override.key.len(),
            });
        }

        for (slot, &byte) in kv_override.key.iter_mut().zip(key_with_nul) {
            *slot = c_char::from_ne_bytes([byte]);
        }

        kv_override.tag = value.tag();
        kv_override.__bindgen_anon_1 = value.value();

        self.push_kv_override_terminator();

        Ok(())
    }

    #[must_use]
    pub const fn kv_overrides(&self) -> KvOverrides<'_> {
        KvOverrides::new(&self.params)
    }

    fn push_kv_override_terminator(mut self: Pin<&mut Self>) {
        self.params.kv_overrides = null();

        self.kv_overrides.push(EMPTY_KV_OVERRIDE);

        self.params.kv_overrides = self.kv_overrides.as_ptr();
    }
}

impl LlamaModelParams {
    /// # Errors
    /// Returns [`ParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_buft_override(mut self: Pin<&mut Self>, key: &CStr) -> Result<(), ParamsError> {
        let buft_override = self
            .buft_overrides
            .get_mut(0)
            .ok_or(ParamsError::NoAvailableOverrideSlot)?;

        if !buft_override.pattern.is_null() {
            return Err(ParamsError::OverrideSlotNotEmpty);
        }

        let pattern = key.to_owned();

        buft_override.pattern = pattern.as_ptr();
        buft_override.buft = unsafe { llama_cpp_bindings_sys::ggml_backend_cpu_buffer_type() };

        self.buft_override_patterns.push(pattern);
        self.push_buft_override_terminator();

        Ok(())
    }

    /// # Errors
    /// Returns [`ParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_moe_override(self: Pin<&mut Self>) -> Result<(), ParamsError> {
        self.add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps")
    }

    fn push_buft_override_terminator(mut self: Pin<&mut Self>) {
        self.params.tensor_buft_overrides = null();

        self.buft_overrides
            .push(llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: ptr::null_mut(),
            });

        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();
    }
}

impl LlamaModelParams {
    /// # Errors
    /// Returns `ParamsError::BackendDeviceNotFound` if any device index is invalid.
    pub fn with_devices(mut self, devices: &[usize]) -> Result<Self, ParamsError> {
        for dev in self.devices.iter_mut() {
            *dev = ptr::null_mut();
        }
        let max_devices = max_devices();
        if devices.len() > max_devices {
            return Err(ParamsError::MaxDevicesExceeded(max_devices));
        }
        for (device_slot, &dev) in self.devices.iter_mut().zip(devices) {
            if dev >= unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() } {
                return Err(ParamsError::BackendDeviceNotFound(dev));
            }
            *device_slot = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(dev) };
        }
        self.params.devices = self.devices.as_mut_ptr();

        Ok(self)
    }

    #[must_use]
    pub fn with_lazy_mode(mut self, lazy_mode: LlamaLazyMode) -> Self {
        self.params.lazy_mode = lazy_mode.into();
        self
    }

    #[must_use]
    pub fn with_load_mode(mut self, load_mode: LlamaLoadMode) -> Self {
        self.params.load_mode = load_mode.into();
        self
    }

    #[must_use]
    pub const fn with_load_mtp(mut self, load_mtp: bool) -> Self {
        self.params.load_mtp = load_mtp;
        self
    }

    #[must_use]
    pub const fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    #[must_use]
    pub const fn with_n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    #[must_use]
    pub const fn with_no_alloc(mut self, no_alloc: bool) -> Self {
        self.params.no_alloc = no_alloc;
        self
    }

    #[must_use]
    pub fn with_split_mode(mut self, split_mode: LlamaSplitMode) -> Self {
        self.params.split_mode = split_mode.into();
        self
    }

    #[must_use]
    pub const fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    #[must_use]
    pub fn devices(&self) -> Vec<usize> {
        self.devices
            .iter()
            .take_while(|device| !device.is_null())
            .filter_map(|device| backend_device_index(*device))
            .collect()
    }

    /// # Errors
    /// Returns [`ParamsError::UnknownLazyMode`] when llama.cpp returns an unknown lazy mode.
    pub fn lazy_mode(&self) -> Result<LlamaLazyMode, ParamsError> {
        LlamaLazyMode::try_from(self.params.lazy_mode)
    }

    /// # Errors
    /// Returns [`ParamsError::UnknownLoadMode`] when llama.cpp returns an unknown load mode.
    pub fn load_mode(&self) -> Result<LlamaLoadMode, ParamsError> {
        LlamaLoadMode::try_from(self.params.load_mode)
    }

    #[must_use]
    pub const fn load_mtp(&self) -> bool {
        self.params.load_mtp
    }

    #[must_use]
    pub const fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    #[must_use]
    pub const fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    #[must_use]
    pub const fn no_alloc(&self) -> bool {
        self.params.no_alloc
    }

    /// # Errors
    /// Returns [`ParamsError::UnknownSplitMode`] if the unknown split mode is encountered.
    pub fn split_mode(&self) -> Result<LlamaSplitMode, ParamsError> {
        LlamaSplitMode::try_from(self.params.split_mode)
    }

    #[must_use]
    pub const fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }
}

impl LlamaModelParams {
    /// # Errors
    ///
    /// Returns one of the [`ParamsError`] variants matching the llama.cpp wrapper's status code.
    pub fn fit_params(
        mut self: Pin<&mut Self>,
        model_path: &CStr,
        LlamaContextParams { context_params, .. }: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
        mut extra: Option<&mut FitExtraModel<'_>>,
        log_level: llama_cpp_bindings_sys::ggml_log_level,
    ) -> Result<FitResult, ParamsError> {
        let max_devices = unsafe { llama_cpp_bindings_sys::llama_max_devices() };
        let max_buft = unsafe { llama_cpp_bindings_sys::llama_max_tensor_buft_overrides() };

        if margins.len() != max_devices {
            return Err(ParamsError::FitMarginsDeviceCountMismatch {
                expected: max_devices,
                actual: margins.len(),
            });
        }

        self.tensor_split.clear();
        self.tensor_split.resize(max_devices, 0.0);

        self.buft_overrides.clear();
        self.buft_overrides.resize(
            max_buft.saturating_add(1),
            llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: ptr::null_mut(),
            },
        );

        self.params.tensor_split = null::<f32>();
        self.params.tensor_buft_overrides = null();

        let mut out_unrecognized_status_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let mut extra_ffi = extra.as_mut().map(|extra_model| extra_model.as_ffi());
        let extra_ptr = extra_ffi
            .as_mut()
            .map_or_else(null, |extra_model| &raw const *extra_model);

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_fit_params(
                model_path.as_ptr(),
                &raw mut self.params,
                &raw mut *context_params,
                self.tensor_split.as_mut_ptr(),
                self.buft_overrides.as_mut_ptr(),
                margins.as_mut_ptr(),
                n_ctx_min,
                extra_ptr,
                log_level,
                &raw mut out_unrecognized_status_code,
                &raw mut out_error,
            )
        };

        fit_params_status_to_result(status, out_unrecognized_status_code, out_error)?;

        self.params.tensor_split = self.tensor_split.as_ptr();
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        Ok(FitResult {
            n_ctx: context_params.n_ctx,
        })
    }
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params = unsafe { llama_cpp_bindings_sys::llama_model_default_params() };
        Self {
            params: default_params,
            kv_overrides: vec![EMPTY_KV_OVERRIDE],
            buft_overrides: vec![llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: null(),
                buft: ptr::null_mut(),
            }],
            buft_override_patterns: Vec::new(),
            devices: Pin::new(
                vec![ptr::null_mut(); max_devices().saturating_add(1)].into_boxed_slice(),
            ),
            tensor_split: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ffi::CString;
    use std::pin::pin;
    use std::ptr;

    use llama_cpp_backend::llama_backend::LlamaBackend;
    use llama_cpp_backend::max_devices::max_devices;
    use llama_cpp_ffi_status::ffi_error::FfiError;

    use super::LlamaModelParams;
    use super::fit_params_status_to_result;
    use crate::constants::KV_OVERRIDE_TEXT_CAPACITY;
    use crate::kv_override_entry::KvOverrideEntry;
    use crate::llama_context_params::LlamaContextParams;
    use crate::llama_lazy_mode::LlamaLazyMode;
    use crate::llama_load_mode::LlamaLoadMode;
    use crate::llama_split_mode::LlamaSplitMode;
    use crate::param_override_value::ParamOverrideValue;
    use crate::params_error::ParamsError;

    fn contract_error_reported_for(
        status: llama_cpp_bindings_sys::llama_rs_fit_params_status,
    ) -> Option<ParamsError> {
        fit_params_status_to_result(status, 0, ptr::null_mut()).err()
    }

    fn contract_error_with(detail: &'static str) -> ParamsError {
        FfiError::ContractViolated {
            operation: "llama_rs_fit_params",
            detail,
        }
        .into()
    }

    #[test]
    fn default_params_match_the_llama_cpp_model_defaults() {
        let params = LlamaModelParams::default();

        assert_eq!(params.n_gpu_layers(), -1);
        assert_eq!(params.main_gpu(), 0);
        assert!(!params.vocab_only());
        assert_eq!(params.load_mode(), Ok(LlamaLoadMode::Auto));
        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer));
        assert_eq!(params.devices(), Vec::<usize>::new());
    }

    #[test]
    fn default_params_use_automatic_lazy_loading() {
        let params = LlamaModelParams::default();

        assert_eq!(params.lazy_mode(), Ok(LlamaLazyMode::Auto));
    }

    #[test]
    fn with_lazy_mode_sets_each_supported_mode() {
        for mode in [LlamaLazyMode::Off, LlamaLazyMode::Auto, LlamaLazyMode::On] {
            let params = LlamaModelParams::default().with_lazy_mode(mode);

            assert_eq!(params.lazy_mode(), Ok(mode));
        }
    }

    #[test]
    fn unknown_lazy_mode_in_model_params_preserves_its_value() {
        let mut params = LlamaModelParams::default();
        let unknown = llama_cpp_bindings_sys::llama_lazy_mode::MAX;
        params.params.lazy_mode = unknown;

        assert_eq!(
            params.lazy_mode(),
            Err(ParamsError::UnknownLazyMode {
                value: i64::from(unknown)
            })
        );
    }

    #[test]
    fn with_n_gpu_layers_sets_the_offload_count() {
        let params = LlamaModelParams::default().with_n_gpu_layers(999);

        assert_eq!(params.n_gpu_layers(), 999);
    }

    #[test]
    fn with_main_gpu_sets_value() {
        let params = LlamaModelParams::default().with_main_gpu(2);

        assert_eq!(params.main_gpu(), 2);
    }

    #[test]
    fn with_split_mode_none_disables_splitting() {
        let params = LlamaModelParams::default().with_split_mode(LlamaSplitMode::None);

        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::None));
    }

    #[test]
    fn with_vocab_only_enables() {
        let params = LlamaModelParams::default().with_vocab_only(true);

        assert!(params.vocab_only());
    }

    #[test]
    fn with_vocab_only_false_disables_vocab_only_loading() {
        let params = LlamaModelParams::default()
            .with_vocab_only(true)
            .with_vocab_only(false);

        assert!(!params.vocab_only());
    }

    #[test]
    fn with_no_alloc_enables() {
        let params = LlamaModelParams::default().with_no_alloc(true);

        assert!(params.no_alloc());
    }

    #[test]
    fn with_no_alloc_false_reenables_allocation() {
        let params = LlamaModelParams::default()
            .with_no_alloc(true)
            .with_no_alloc(false);

        assert!(!params.no_alloc());
    }

    #[test]
    fn with_no_alloc_preserves_load_mode() {
        let params = LlamaModelParams::default()
            .with_load_mode(LlamaLoadMode::Mmap)
            .with_no_alloc(true);

        assert!(params.no_alloc());
        assert_eq!(params.load_mode(), Ok(LlamaLoadMode::Mmap));
    }

    #[test]
    fn with_load_mtp_enables_mtp_loading() {
        let params = LlamaModelParams::default().with_load_mtp(true);

        assert!(params.load_mtp());
    }

    #[test]
    fn with_load_mtp_false_disables_mtp_loading() {
        let params = LlamaModelParams::default()
            .with_load_mtp(true)
            .with_load_mtp(false);

        assert!(!params.load_mtp());
    }

    #[test]
    fn default_no_alloc_is_false() {
        let params = LlamaModelParams::default();

        assert!(!params.no_alloc());
    }

    #[test]
    fn debug_output_lists_every_field_including_the_lazy_mode() {
        let params = LlamaModelParams::default().with_lazy_mode(LlamaLazyMode::On);

        assert_eq!(
            format!("{params:?}"),
            "LlamaModelParams { n_gpu_layers: -1, main_gpu: 0, vocab_only: false, \
             lazy_mode: Ok(On), load_mode: Ok(Auto), load_mtp: false, \
             split_mode: Ok(Layer), devices: [], \
             kv_overrides: \"vec of kv_overrides\", .. }"
        );
    }

    #[test]
    fn builder_chaining_preserves_all_values() {
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(10)
            .with_main_gpu(1)
            .with_split_mode(LlamaSplitMode::Row)
            .with_vocab_only(true)
            .with_load_mode(LlamaLoadMode::MmapMlock);

        assert_eq!(params.n_gpu_layers(), 10);
        assert_eq!(params.main_gpu(), 1);
        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Row));
        assert!(params.vocab_only());
        assert_eq!(params.load_mode(), Ok(LlamaLoadMode::MmapMlock));
    }

    #[test]
    fn with_devices_accepts_an_empty_device_list() {
        assert_eq!(
            LlamaModelParams::default()
                .with_devices(&[])
                .map(|params| params.devices()),
            Ok(Vec::new())
        );
    }

    #[test]
    fn add_cpu_buft_override_fills_an_empty_override_slot() {
        let mut params = pin!(LlamaModelParams::default());

        assert_eq!(
            params.as_mut().add_cpu_buft_override(c"test_pattern"),
            Ok(())
        );
        assert_eq!(
            unsafe { CStr::from_ptr(params.buft_overrides.first().unwrap().pattern) },
            c"test_pattern"
        );
        assert_eq!(params.buft_overrides.first().unwrap().buft, unsafe {
            llama_cpp_bindings_sys::ggml_backend_cpu_buffer_type()
        });
        assert_eq!(params.buft_overrides.get(1).unwrap().pattern, ptr::null());
        assert_eq!(
            params.params.tensor_buft_overrides,
            params.buft_overrides.as_ptr()
        );
    }

    #[test]
    fn a_buft_override_pattern_stays_valid_after_the_callers_buffer_changes() {
        let mut caller_buffer = b"blk\\.0\\.ffn\0".to_vec();
        let mut params = pin!(LlamaModelParams::default());

        params
            .as_mut()
            .add_cpu_buft_override(CStr::from_bytes_with_nul(&caller_buffer).unwrap())
            .unwrap();
        caller_buffer.fill(b'x');

        assert_eq!(
            unsafe { CStr::from_ptr(params.buft_overrides.first().unwrap().pattern) },
            c"blk\\.0\\.ffn"
        );
    }

    #[test]
    fn add_cpu_buft_override_twice_fails_with_slot_not_empty() {
        let mut params = pin!(LlamaModelParams::default());
        params
            .as_mut()
            .add_cpu_buft_override(c"first_pattern")
            .unwrap();
        let result = params.as_mut().add_cpu_buft_override(c"second_pattern");

        assert_eq!(result.unwrap_err(), ParamsError::OverrideSlotNotEmpty);
    }

    #[test]
    fn add_cpu_moe_override_keeps_expert_tensors_in_cpu_memory() {
        let mut params = pin!(LlamaModelParams::default());

        assert_eq!(params.as_mut().add_cpu_moe_override(), Ok(()));
        assert_eq!(
            unsafe { CStr::from_ptr(params.buft_overrides.first().unwrap().pattern) },
            c"\\.ffn_(up|down|gate)_(ch|)exps"
        );
        assert_eq!(params.buft_overrides.first().unwrap().buft, unsafe {
            llama_cpp_bindings_sys::ggml_backend_cpu_buffer_type()
        });
    }

    #[test]
    fn append_kv_override_twice_fails_with_slot_not_empty() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("first_key").unwrap();
        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(1))
            .unwrap();

        let key2 = CString::new("second_key").unwrap();
        let result = params
            .as_mut()
            .append_kv_override(&key2, ParamOverrideValue::Int(2));

        assert_eq!(result.unwrap_err(), ParamsError::OverrideSlotNotEmpty);
    }

    #[test]
    fn more_devices_than_llama_cpp_supports_are_refused() {
        let too_many = vec![0; max_devices() + 1];

        assert_eq!(
            LlamaModelParams::default().with_devices(&too_many).err(),
            Some(ParamsError::MaxDevicesExceeded(max_devices()))
        );
    }

    #[test]
    fn a_full_device_list_stays_null_terminated() {
        let params = LlamaModelParams::default()
            .with_devices(&vec![0; max_devices()])
            .unwrap();

        assert_eq!(params.devices.len(), max_devices() + 1);
        assert_eq!(params.devices.last(), Some(&ptr::null_mut()));
    }

    #[test]
    fn a_kv_override_key_of_any_bytes_is_stored_as_given() {
        let key = c"cl\xc3\xa9";
        let mut params = pin!(LlamaModelParams::default());

        params
            .as_mut()
            .append_kv_override(key, ParamOverrideValue::Int(1))
            .unwrap();

        assert_eq!(
            unsafe { CStr::from_ptr(params.kv_overrides.first().unwrap().key.as_ptr()) },
            key
        );
    }

    #[test]
    fn a_kv_override_key_longer_than_llama_cpp_holds_is_refused() {
        let key = CString::new("k".repeat(KV_OVERRIDE_TEXT_CAPACITY)).unwrap();
        let mut params = pin!(LlamaModelParams::default());

        assert_eq!(
            params
                .as_mut()
                .append_kv_override(&key, ParamOverrideValue::Int(1)),
            Err(ParamsError::OverrideKeyTooLong {
                length_with_nul: KV_OVERRIDE_TEXT_CAPACITY + 1,
                capacity: KV_OVERRIDE_TEXT_CAPACITY,
            })
        );
    }

    #[test]
    fn with_devices_sets_devices_when_available() {
        #[cfg(feature = "dynamic-backends")]
        unsafe {
            llama_cpp_bindings_sys::ggml_backend_load_all();
        }

        let dev_count = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() };
        assert!(dev_count > 0, "Test requires at least one backend device");

        let params = LlamaModelParams::default().with_devices(&[0]).unwrap();

        assert_eq!(params.devices(), vec![0]);
    }

    #[test]
    fn with_devices_invalid_index_returns_not_found() {
        let invalid_index = usize::MAX;
        let result = LlamaModelParams::default().with_devices(&[invalid_index]);

        assert!(
            matches!(
                result,
                Err(ParamsError::BackendDeviceNotFound(device)) if device == invalid_index
            ),
            "an out-of-range device index must not be found, got {result:?}"
        );
    }

    #[test]
    fn append_kv_override_with_empty_slot_vector_returns_no_available_slot() {
        let mut params = LlamaModelParams::default();
        params.kv_overrides.clear();
        let mut pinned = pin!(params);

        let result = pinned
            .as_mut()
            .append_kv_override(c"any_key", ParamOverrideValue::Int(1));

        assert_eq!(result.unwrap_err(), ParamsError::NoAvailableOverrideSlot);
    }

    #[test]
    fn add_cpu_buft_override_with_empty_slot_vector_returns_no_available_slot() {
        let mut params = LlamaModelParams::default();
        params.buft_overrides.clear();
        let mut pinned = pin!(params);

        let result = pinned.as_mut().add_cpu_buft_override(c"any_pattern");

        assert_eq!(result.unwrap_err(), ParamsError::NoAvailableOverrideSlot);
    }

    #[test]
    fn fit_params_reports_a_missing_model_file_as_aborted() {
        let _backend = LlamaBackend::init();
        let mut params = pin!(LlamaModelParams::default());
        let mut context_params = LlamaContextParams::default();
        let mut margins = vec![0usize; max_devices()];

        let bogus_path = c"/nonexistent/path/to/model.gguf";
        let result = params.as_mut().fit_params(
            bogus_path,
            &mut context_params,
            &mut margins,
            512,
            None,
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
        );

        assert_eq!(result.err(), Some(ParamsError::FitAborted));
    }

    #[test]
    fn fit_params_refuses_margins_not_sized_to_the_device_count() {
        let mut params = pin!(LlamaModelParams::default());
        let mut context_params = LlamaContextParams::default();
        let mut margins = vec![0usize; max_devices() + 1];

        let result = params.as_mut().fit_params(
            c"/nonexistent/path/to/model.gguf",
            &mut context_params,
            &mut margins,
            512,
            None,
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
        );

        assert_eq!(
            result.err(),
            Some(ParamsError::FitMarginsDeviceCountMismatch {
                expected: max_devices(),
                actual: max_devices() + 1,
            })
        );
    }

    #[test]
    fn fit_params_status_ok_returns_ok() {
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_OK,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn fit_params_status_reported_failure_returns_no_fitting_memory_layout() {
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_FAILURE,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(ParamsError::NoFittingMemoryLayout));
    }

    #[test]
    fn fit_params_status_reported_error_returns_aborted() {
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_ERROR,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(ParamsError::FitAborted));
    }

    #[test]
    fn fit_params_status_unrecognized_code_returns_unknown_status() {
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_RETURNED_UNRECOGNIZED_STATUS_CODE,
            42,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(ParamsError::UnknownFitStatus { code: 42 }));
    }

    #[test]
    fn fit_params_status_allocation_failed_returns_not_enough_memory() {
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(ParamsError::FitNotEnoughMemory));
    }

    #[test]
    fn fit_params_status_cxx_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"fitting threw".as_ptr()) };
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            out_error,
        );

        assert_eq!(
            result,
            Err(ParamsError::FitReported {
                message: "fitting threw".to_owned()
            })
        );
    }

    #[test]
    fn fit_params_status_reports_an_exception_without_a_message_as_a_contract_violation() {
        let result = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(FfiError::ContractViolated {
                operation: "llama_rs_fit_params",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn fit_params_unknown_wrapper_status_is_preserved() {
        let result = fit_params_status_to_result(255, 0, ptr::null_mut());

        assert_eq!(
            result,
            Err(ParamsError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_fit_params",
                code: 255,
            }))
        );
    }

    #[test]
    fn fit_params_status_llama_cpp_out_of_memory_returns_llama_cpp_out_of_memory() {
        let outcome = fit_params_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );

        assert_eq!(outcome.err(), Some(ParamsError::FitLlamaCppOutOfMemory));
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_path_model_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_PATH_MODEL_ARG
            ),
            Some(contract_error_with("was given a null path_model argument"))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_mparams_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_MPARAMS_ARG
            ),
            Some(contract_error_with("was given a null mparams argument"))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_cparams_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_CPARAMS_ARG
            ),
            Some(contract_error_with("was given a null cparams argument"))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_out_unrecognized_status_code_arg_as_a_contract_violation()
     {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_OUT_UNRECOGNIZED_STATUS_CODE_ARG
            ),
            Some(contract_error_with(
                "was given a null out_unrecognized_status_code argument"
            ))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_out_error_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_OUT_ERROR_ARG
            ),
            Some(contract_error_with("was given a null out_error argument"))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_extra_path_model_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_EXTRA_PATH_MODEL_ARG
            ),
            Some(contract_error_with(
                "was given an extra model with a null path_model argument"
            ))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_extra_mparams_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_EXTRA_MPARAMS_ARG
            ),
            Some(contract_error_with(
                "was given an extra model with a null mparams argument"
            ))
        );
    }

    #[test]
    fn fit_params_status_to_result_reports_a_null_extra_cparams_arg_as_a_contract_violation() {
        assert_eq!(
            contract_error_reported_for(
                llama_cpp_bindings_sys::LLAMA_RS_FIT_PARAMS_NULL_EXTRA_CPARAMS_ARG
            ),
            Some(contract_error_with(
                "was given an extra model with a null cparams argument"
            ))
        );
    }

    #[test]
    fn kv_overrides_empty_by_default() {
        let params = LlamaModelParams::default();
        let overrides = params.kv_overrides();
        let count = overrides.into_iter().count();

        assert_eq!(count, 0);
    }

    #[test]
    fn kv_overrides_iterates_single_entry() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("test_key").unwrap();

        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(42))
            .unwrap();

        let entries: Result<Vec<_>, _> = params.kv_overrides().into_iter().collect();
        let entries = entries.expect("known override tags must convert");

        assert_eq!(
            entries,
            vec![KvOverrideEntry {
                key: CString::new("test_key").expect("the literal has no nul byte"),
                value: ParamOverrideValue::Int(42),
            }]
        );
    }

    #[test]
    fn kv_overrides_preserves_unknown_tag_error() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("valid_key").unwrap();

        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(99))
            .unwrap();

        params.kv_overrides.first_mut().unwrap().tag = 9999;

        let entry = params
            .kv_overrides()
            .into_iter()
            .next()
            .expect("one override must be present");

        assert_eq!(entry.unwrap_err(), ParamsError::UnknownKvOverrideTag(9999));
    }
}
