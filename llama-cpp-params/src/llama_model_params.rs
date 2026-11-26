use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::num::NonZeroU32;
use std::path::Path;
use std::pin::Pin;
use std::ptr;
use std::ptr::null;

use llama_cpp_backend::backend_device_index::backend_device_index;
use llama_cpp_backend::max_devices::max_devices;
use llama_cpp_bindings_sys::bindings::llama_rs_fit_model_traits;
use llama_cpp_bindings_sys::bindings::llama_rs_fit_model_traits_status;
use llama_cpp_bindings_sys::bindings::llama_rs_fit_params_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;
use llama_cpp_path_c_string::path_c_string::path_c_string;

use crate::constants::EMPTY_KV_OVERRIDE;
use crate::fit_extra_model::FitExtraModel;
use crate::fit_result::FitResult;
use crate::kv_overrides::KvOverrides;
use crate::llama_context_params::LlamaContextParams;
use crate::llama_lazy_mode::LlamaLazyMode;
use crate::llama_load_mode::LlamaLoadMode;
use crate::llama_split_mode::LlamaSplitMode;
use crate::model_output_traits::ModelOutputTraits;
use crate::param_override_value::ParamOverrideValue;
use crate::params_error::ParamsError;
use crate::sequence_output::SequenceOutput;
use crate::sequence_output_capacity::SequenceOutputCapacity;

/// # Safety
///
/// On `LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message
/// the C++ side allocated; it is freed here.
unsafe fn fit_params_status_to_result(
    status: llama_rs_fit_params_status,
    out_unrecognized_status_code: i32,
    out_error: *mut c_char,
) -> Result<(), ParamsError> {
    match status {
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_OK => Ok(()),
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_FAILURE => {
            Err(ParamsError::NoFittingMemoryLayout)
        }
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_ERROR => {
            Err(ParamsError::FitAborted)
        }
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_RETURNED_UNRECOGNIZED_STATUS_CODE => {
            Err(ParamsError::UnknownFitStatus {
                code: out_unrecognized_status_code,
            })
        }
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParamsError::FitNotEnoughMemory)
        }
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParamsError::FitLlamaCppOutOfMemory)
        }
        llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_fit_params", |message| {
                    ParamsError::FitReported { message }
                }))
        }
    }
}

fn prepare_sequences_llama_cpp_can_create(
    model_path: &CStr,
    model_params: llama_cpp_bindings_sys::bindings::llama_model_params,
    context_params: &mut LlamaContextParams,
    exceeding_sequences_error: fn(u32, u32) -> ParamsError,
) -> Result<(), ParamsError> {
    let mut model_traits = llama_rs_fit_model_traits {
        causal_attention: false,
        has_encoder: false,
        n_ctx_train: 0,
    };

    match unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_fit_model_traits_of(
            model_path.as_ptr(),
            model_params,
            &raw mut model_traits,
        )
    } {
        llama_rs_fit_model_traits_status::LLAMA_RS_FIT_MODEL_TRAITS_READ => {
            let model_output_traits = ModelOutputTraits::from(model_traits);

            *context_params = model_output_traits.context_params(*context_params);

            match (SequenceOutputCapacity {
                context_params,
                model_output_traits,
            })
            .sequence_output()?
            {
                SequenceOutput::ExceedsCapacity {
                    n_seq_max,
                    output_capacity,
                } => Err(exceeding_sequences_error(n_seq_max, output_capacity)),
                SequenceOutput::FitsCapacity => Ok(()),
            }
        }
        llama_rs_fit_model_traits_status::LLAMA_RS_FIT_MODEL_TRAITS_MODEL_UNLOADABLE => Ok(()),
    }
}

pub struct LlamaModelParams {
    params: llama_cpp_bindings_sys::bindings::llama_model_params,
    kv_overrides: Vec<llama_cpp_bindings_sys::bindings::llama_model_kv_override>,
    buft_overrides: Vec<llama_cpp_bindings_sys::bindings::llama_model_tensor_buft_override>,
    buft_override_patterns: Vec<CString>,
    devices: Pin<Box<[llama_cpp_bindings_sys::bindings::ggml_backend_dev_t]>>,
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
    /// the slot is not empty, the key is longer than llama.cpp holds, or a string value holds
    /// no NUL.
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

        if let ParamOverrideValue::Str(text) = value
            && !text.contains(&0)
        {
            return Err(ParamsError::OverrideTextNotNulTerminated {
                key: key.to_owned(),
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
        unsafe { KvOverrides::new(&self.params) }
    }

    fn push_kv_override_terminator(mut self: Pin<&mut Self>) {
        self.params.kv_overrides = null();

        self.kv_overrides.push(EMPTY_KV_OVERRIDE);

        self.params.kv_overrides = self.kv_overrides.as_ptr();
    }
}

impl LlamaModelParams {
    /// # Errors
    /// Returns [`ParamsError`] if the internal override vector has no available slot or the
    /// slot is not empty.
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
        buft_override.buft =
            unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_cpu_buffer_type() };

        self.buft_override_patterns.push(pattern);
        self.push_buft_override_terminator();

        Ok(())
    }

    /// # Errors
    /// Returns [`ParamsError`] if the internal override vector has no available slot or the
    /// slot is not empty.
    pub fn add_cpu_moe_override(self: Pin<&mut Self>) -> Result<(), ParamsError> {
        self.add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps")
    }

    fn push_buft_override_terminator(mut self: Pin<&mut Self>) {
        self.params.tensor_buft_overrides = null();

        self.buft_overrides.push(
            llama_cpp_bindings_sys::bindings::llama_model_tensor_buft_override {
                pattern: null(),
                buft: ptr::null_mut(),
            },
        );

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
            if dev >= unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_dev_count() } {
                return Err(ParamsError::BackendDeviceNotFound(dev));
            }
            *device_slot = unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_dev_get(dev) };
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

    /// A copy of the llama.cpp model parameters, for a llama.cpp call that takes them by value.
    ///
    /// The copy points into buffers this value owns, such as its devices and overrides, so it is
    /// valid only while `self` stays borrowed: hand it to llama.cpp within the call and never
    /// keep it.
    #[must_use]
    pub const fn as_ffi(&self) -> llama_cpp_bindings_sys::bindings::llama_model_params {
        self.params
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
    /// Returns [`ParamsError::FitModelPathInvalid`] or [`ParamsError::FitExtraModelPathInvalid`]
    /// when a model path cannot be passed to llama.cpp,
    /// [`ParamsError::FitMarginsDeviceCountMismatch`] when `margins` does not hold one entry per
    /// device, [`ParamsError::FitSequencesExceedOutputCapacity`] or
    /// [`ParamsError::FitExtraModelSequencesExceedOutputCapacity`] when llama.cpp could not give
    /// every sequence of the measured context an output, [`ParamsError::UnknownAttentionType`]
    /// when the context params hold an attention type llama.cpp does not define, or the variant
    /// naming how llama.cpp failed to fit the parameters.
    pub fn fit_params(
        self: Pin<&mut Self>,
        model_path: &Path,
        context_params: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
        mut extra: Option<&mut FitExtraModel<'_, Self>>,
        log_level: llama_cpp_bindings_sys::bindings::ggml_log_level,
    ) -> Result<FitResult, ParamsError> {
        let max_devices = unsafe { llama_cpp_bindings_sys::bindings::llama_max_devices() };
        let max_buft =
            unsafe { llama_cpp_bindings_sys::bindings::llama_max_tensor_buft_overrides() };

        let model_path = path_c_string(model_path).map_err(ParamsError::FitModelPathInvalid)?;
        let extra_model_path = extra
            .as_ref()
            .map(|extra_model| path_c_string(extra_model.model_path))
            .transpose()
            .map_err(ParamsError::FitExtraModelPathInvalid)?;

        if margins.len() != max_devices {
            return Err(ParamsError::FitMarginsDeviceCountMismatch {
                expected: max_devices,
                actual: margins.len(),
            });
        }

        prepare_sequences_llama_cpp_can_create(
            &model_path,
            self.params,
            context_params,
            |n_seq_max, output_capacity| ParamsError::FitSequencesExceedOutputCapacity {
                n_seq_max,
                output_capacity,
            },
        )?;

        if let Some((extra_model, extra_model_path)) = extra.as_mut().zip(extra_model_path.as_ref())
        {
            prepare_sequences_llama_cpp_can_create(
                extra_model_path,
                extra_model.model_params.params,
                extra_model.context_params,
                |n_seq_max, output_capacity| {
                    ParamsError::FitExtraModelSequencesExceedOutputCapacity {
                        n_seq_max,
                        output_capacity,
                    }
                },
            )?;
        }

        let Self {
            params,
            buft_overrides,
            tensor_split,
            ..
        } = self.get_mut();

        tensor_split.clear();
        tensor_split.resize(max_devices, 0.0);

        buft_overrides.clear();
        buft_overrides.resize(
            max_buft.saturating_add(1),
            llama_cpp_bindings_sys::bindings::llama_model_tensor_buft_override {
                pattern: null(),
                buft: ptr::null_mut(),
            },
        );

        params.tensor_split = null::<f32>();
        params.tensor_buft_overrides = null();

        let mut out_unrecognized_status_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let mut extra_ffi = extra.zip(extra_model_path.as_ref()).map(
            |(
                FitExtraModel {
                    model_params: extra_model_params,
                    context_params: extra_context_params,
                    shares_model,
                    ..
                },
                extra_model_path,
            )| llama_cpp_bindings_sys::bindings::llama_rs_fit_extra_model {
                path_model: extra_model_path.as_ptr(),
                mparams: &raw mut extra_model_params.params,
                cparams: extra_context_params.as_mut_ffi(),
                shares_model: *shares_model,
            },
        );
        let extra_ptr = extra_ffi
            .as_mut()
            .map_or_else(null, |extra_model| &raw const *extra_model);

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_fit_params(
                model_path.as_ptr(),
                &raw mut *params,
                context_params.as_mut_ffi(),
                tensor_split.as_mut_ptr(),
                buft_overrides.as_mut_ptr(),
                margins.as_mut_ptr(),
                n_ctx_min,
                extra_ptr,
                log_level,
                &raw mut out_unrecognized_status_code,
                &raw mut out_error,
            )
        };

        unsafe { fit_params_status_to_result(status, out_unrecognized_status_code, out_error) }?;

        params.tensor_split = tensor_split.as_ptr();
        params.tensor_buft_overrides = buft_overrides.as_ptr();

        Ok(FitResult {
            n_ctx: context_params.n_ctx().map_or(0, NonZeroU32::get),
        })
    }
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params =
            unsafe { llama_cpp_bindings_sys::bindings::llama_model_default_params() };
        Self {
            params: default_params,
            kv_overrides: vec![EMPTY_KV_OVERRIDE],
            buft_overrides: vec![
                llama_cpp_bindings_sys::bindings::llama_model_tensor_buft_override {
                    pattern: null(),
                    buft: ptr::null_mut(),
                },
            ],
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
    use std::path::Path;
    use std::pin::pin;
    use std::ptr;

    use llama_cpp_backend::llama_backend::LlamaBackend;
    use llama_cpp_backend::max_devices::max_devices;
    use llama_cpp_bindings_sys::bindings::llama_rs_fit_params_status;

    use super::LlamaModelParams;
    use super::fit_params_status_to_result;
    use crate::constants::KV_OVERRIDE_TEXT_CAPACITY;
    use crate::fit_extra_model::FitExtraModel;
    use crate::kv_override_entry::KvOverrideEntry;
    use crate::llama_context_params::LlamaContextParams;
    use crate::llama_lazy_mode::LlamaLazyMode;
    use crate::llama_load_mode::LlamaLoadMode;
    use crate::llama_split_mode::LlamaSplitMode;
    use crate::param_override_value::ParamOverrideValue;
    use crate::params_error::ParamsError;

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
        let unknown = llama_cpp_bindings_sys::bindings::llama_lazy_mode::MAX;
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
            llama_cpp_bindings_sys::bindings::ggml_backend_cpu_buffer_type()
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
            llama_cpp_bindings_sys::bindings::ggml_backend_cpu_buffer_type()
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
    fn a_string_override_value_without_a_nul_is_refused() {
        let mut params = pin!(LlamaModelParams::default());

        assert_eq!(
            params.as_mut().append_kv_override(
                c"general.name",
                ParamOverrideValue::Str([b'x'.cast_signed(); KV_OVERRIDE_TEXT_CAPACITY])
            ),
            Err(ParamsError::OverrideTextNotNulTerminated {
                key: c"general.name".to_owned(),
            })
        );
    }

    #[test]
    fn with_devices_sets_devices_when_available() {
        unsafe {
            llama_cpp_bindings_sys::bindings::ggml_backend_load_all();
        }

        let dev_count = unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_dev_count() };
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

        let result = params.as_mut().fit_params(
            Path::new("/nonexistent/path/to/model.gguf"),
            &mut context_params,
            &mut margins,
            512,
            None,
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
        );

        assert_eq!(result.err(), Some(ParamsError::FitAborted));
    }

    #[test]
    fn fit_params_refuses_margins_not_sized_to_the_device_count() {
        let mut params = pin!(LlamaModelParams::default());
        let mut context_params = LlamaContextParams::default();
        let mut margins = vec![0usize; max_devices() + 1];

        let result = params.as_mut().fit_params(
            Path::new("/nonexistent/path/to/model.gguf"),
            &mut context_params,
            &mut margins,
            512,
            None,
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
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
    fn fit_params_refuses_a_model_path_llama_cpp_cannot_read() {
        let mut params = pin!(LlamaModelParams::default());
        let mut context_params = LlamaContextParams::default();
        let mut margins = vec![0usize; max_devices()];

        let result = params.as_mut().fit_params(
            Path::new("/models/deci\0sion.gguf"),
            &mut context_params,
            &mut margins,
            512,
            None,
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
        );

        assert!(matches!(result, Err(ParamsError::FitModelPathInvalid(_))));
    }

    #[test]
    fn fit_params_refuses_an_extra_model_path_llama_cpp_cannot_read() {
        let mut params = pin!(LlamaModelParams::default());
        let mut extra_model_params = LlamaModelParams::default();
        let mut context_params = LlamaContextParams::default();
        let mut extra_context_params = LlamaContextParams::default();
        let mut margins = vec![0usize; max_devices()];

        let result = params.as_mut().fit_params(
            Path::new("/models/target.gguf"),
            &mut context_params,
            &mut margins,
            512,
            Some(&mut FitExtraModel {
                model_path: Path::new("/models/dra\0ft.gguf"),
                model_params: &mut extra_model_params,
                context_params: &mut extra_context_params,
                shares_model: false,
            }),
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
        );

        assert!(matches!(
            result,
            Err(ParamsError::FitExtraModelPathInvalid(_))
        ));
    }

    #[test]
    fn a_successful_fit_status_is_ok() {
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_OK,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn fit_params_status_reported_failure_returns_no_fitting_memory_layout() {
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_FAILURE,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ParamsError::NoFittingMemoryLayout));
    }

    #[test]
    fn fit_params_status_reported_error_returns_aborted() {
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_ERROR,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ParamsError::FitAborted));
    }

    #[test]
    fn fit_params_status_unrecognized_code_returns_unknown_status() {
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_RETURNED_UNRECOGNIZED_STATUS_CODE,
                42,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ParamsError::UnknownFitStatus { code: 42 }));
    }

    #[test]
    fn fit_params_status_allocation_failed_returns_not_enough_memory() {
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ParamsError::FitNotEnoughMemory));
    }

    #[test]
    fn fit_params_status_cxx_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"fitting threw".as_ptr())
        };
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            )
        };

        assert_eq!(
            result,
            Err(ParamsError::FitReported {
                message: "fitting threw".to_owned()
            })
        );
    }

    #[test]
    fn fit_params_status_cxx_exception_with_a_message_that_is_not_utf8_keeps_its_bytes() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"\xff".as_ptr()) };
        let result = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            )
        };

        assert_eq!(
            result,
            Err(ParamsError::ExceptionMessageNotUtf8 {
                operation: "llama_rs_fit_params",
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            })
        );
    }

    #[test]
    fn fit_params_status_llama_cpp_out_of_memory_returns_llama_cpp_out_of_memory() {
        let outcome = unsafe {
            fit_params_status_to_result(
                llama_rs_fit_params_status::LLAMA_RS_FIT_PARAMS_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(outcome.err(), Some(ParamsError::FitLlamaCppOutOfMemory));
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
