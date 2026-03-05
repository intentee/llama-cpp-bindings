//! A safe wrapper around `llama_model_params`.

use crate::LlamaCppError;
use crate::error::ModelParamsError;
use crate::model::params::kv_overrides::KvOverrides;
use std::ffi::{CStr, c_char};
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::ptr::null;

pub mod kv_overrides;

const LLAMA_SPLIT_MODE_NONE: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_NONE as i8;
const LLAMA_SPLIT_MODE_LAYER: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_LAYER as i8;
const LLAMA_SPLIT_MODE_ROW: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_ROW as i8;

/// A rusty wrapper around `llama_split_mode`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaSplitMode {
    /// Single GPU
    None = LLAMA_SPLIT_MODE_NONE,
    /// Split layers and KV across GPUs
    Layer = LLAMA_SPLIT_MODE_LAYER,
    /// Split layers and KV across GPUs, use tensor parallelism if supported
    Row = LLAMA_SPLIT_MODE_ROW,
}

/// An error that occurs when unknown split mode is encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaSplitModeParseError(pub i32);

/// Create a `LlamaSplitMode` from a `i32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<i32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|_| LlamaSplitModeParseError(value))?;
        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            _ => Err(LlamaSplitModeParseError(value)),
        }
    }
}

/// Create a `LlamaSplitMode` from a `u32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<u32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|_| LlamaSplitModeParseError(value.try_into().unwrap_or(i32::MAX)))?;
        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            _ => Err(LlamaSplitModeParseError(
                value.try_into().unwrap_or(i32::MAX),
            )),
        }
    }
}

/// Create a `i32` from a `LlamaSplitMode`.
impl From<LlamaSplitMode> for i32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE.into(),
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER.into(),
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW.into(),
        }
    }
}

/// Create a `u32` from a `LlamaSplitMode`.
impl From<LlamaSplitMode> for u32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE as u32,
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER as u32,
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW as u32,
        }
    }
}

/// The default split mode is `Layer` in llama.cpp.
impl Default for LlamaSplitMode {
    fn default() -> Self {
        LlamaSplitMode::Layer
    }
}

/// The maximum number of devices supported.
///
/// The real maximum number of devices is the lesser one of this value and the value returned by
/// `llama_cpp_bindings::max_devices()`.
pub const LLAMA_CPP_MAX_DEVICES: usize = 16;

/// A safe wrapper around `llama_model_params`.
pub struct LlamaModelParams {
    /// The underlying `llama_model_params` from the C API.
    pub params: llama_cpp_bindings_sys::llama_model_params,
    kv_overrides: Vec<llama_cpp_bindings_sys::llama_model_kv_override>,
    buft_overrides: Vec<llama_cpp_bindings_sys::llama_model_tensor_buft_override>,
    devices: Pin<Box<[llama_cpp_bindings_sys::ggml_backend_dev_t; LLAMA_CPP_MAX_DEVICES]>>,
}

impl Debug for LlamaModelParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModelParams")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("use_mmap", &self.params.use_mmap)
            .field("use_mlock", &self.params.use_mlock)
            .field("split_mode", &self.split_mode())
            .field("devices", &self.devices)
            .field("kv_overrides", &"vec of kv_overrides")
            .finish()
    }
}

impl LlamaModelParams {
    /// See [`KvOverrides`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// let params = Box::pin(LlamaModelParams::default());
    /// let kv_overrides = params.kv_overrides();
    /// let count = kv_overrides.into_iter().count();
    /// assert_eq!(count, 0);
    /// ```
    #[must_use]
    pub fn kv_overrides(&self) -> KvOverrides<'_> {
        KvOverrides::new(self)
    }

    /// Appends a key-value override to the model parameters. It must be pinned as this creates a self-referential struct.
    ///
    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::ffi::{CStr, CString};
    /// use std::pin::pin;
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// # use llama_cpp_bindings::model::params::kv_overrides::ParamOverrideValue;
    /// let mut params = pin!(LlamaModelParams::default());
    /// let key = CString::new("key").expect("CString::new failed");
    /// params.as_mut().append_kv_override(&key, ParamOverrideValue::Int(50)).unwrap();
    ///
    /// let kv_overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
    /// assert_eq!(kv_overrides.len(), 1);
    ///
    /// let (k, v) = &kv_overrides[0];
    /// assert_eq!(v, &ParamOverrideValue::Int(50));
    ///
    /// assert_eq!(k.to_bytes(), b"key", "expected key to be 'key', was {:?}", k);
    /// ```
    pub fn append_kv_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
        value: kv_overrides::ParamOverrideValue,
    ) -> Result<(), ModelParamsError> {
        let kv_override = self
            .kv_overrides
            .get_mut(0)
            .ok_or(ModelParamsError::NoAvailableSlot)?;

        if kv_override.key[0] != 0 {
            return Err(ModelParamsError::SlotNotEmpty);
        }

        for (i, &byte) in key.to_bytes_with_nul().iter().enumerate() {
            kv_override.key[i] = c_char::try_from(byte)
                .map_err(|_| ModelParamsError::InvalidCharacterInKey(byte))?;
        }

        kv_override.tag = value.tag();
        kv_override.__bindgen_anon_1 = value.value();

        // set to null pointer for panic safety (as push may move the vector, invalidating the pointer)
        self.params.kv_overrides = null();

        // push the next one to ensure we maintain the iterator invariant of ending with a 0
        self.kv_overrides
            .push(llama_cpp_bindings_sys::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            });

        // set the pointer to the (potentially) new vector
        self.params.kv_overrides = self.kv_overrides.as_ptr();

        Ok(())
    }
}

impl LlamaModelParams {
    /// Adds buffer type overrides to move all mixture-of-experts layers to CPU.
    ///
    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_moe_override(self: Pin<&mut Self>) -> Result<(), ModelParamsError> {
        self.add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps")
    }

    /// Appends a buffer type override to the model parameters, to move layers matching pattern to CPU.
    /// It must be pinned as this creates a self-referential struct.
    ///
    /// # Errors
    /// Returns [`ModelParamsError`] if the internal override vector has no available slot,
    /// the slot is not empty, or the key contains invalid characters.
    pub fn add_cpu_buft_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
    ) -> Result<(), ModelParamsError> {
        let buft_override = self
            .buft_overrides
            .get_mut(0)
            .ok_or(ModelParamsError::NoAvailableSlot)?;

        if !buft_override.pattern.is_null() {
            return Err(ModelParamsError::SlotNotEmpty);
        }

        for &byte in key.to_bytes_with_nul() {
            c_char::try_from(byte).map_err(|_| ModelParamsError::InvalidCharacterInKey(byte))?;
        }

        buft_override.pattern = key.as_ptr();
        buft_override.buft = unsafe { llama_cpp_bindings_sys::ggml_backend_cpu_buffer_type() };

        // set to null pointer for panic safety (as push may move the vector, invalidating the pointer)
        self.params.tensor_buft_overrides = null();

        // push the next one to ensure we maintain the iterator invariant of ending with a 0
        self.buft_overrides
            .push(llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            });

        // set the pointer to the (potentially) new vector
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        Ok(())
    }
}

impl LlamaModelParams {
    /// Get the number of layers to offload to the GPU.
    #[must_use]
    pub fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    /// The GPU that is used for scratch and small tensors
    #[must_use]
    pub fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    /// only load the vocabulary, no weights
    #[must_use]
    pub fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    /// use mmap if possible
    #[must_use]
    pub fn use_mmap(&self) -> bool {
        self.params.use_mmap
    }

    /// force system to keep model in RAM
    #[must_use]
    pub fn use_mlock(&self) -> bool {
        self.params.use_mlock
    }

    /// get the split mode
    ///
    /// # Errors
    /// Returns `LlamaSplitModeParseError` if the unknown split mode is encountered.
    pub fn split_mode(&self) -> Result<LlamaSplitMode, LlamaSplitModeParseError> {
        LlamaSplitMode::try_from(self.params.split_mode)
    }

    /// get the devices
    #[must_use]
    pub fn devices(&self) -> Vec<usize> {
        let mut backend_devices = Vec::new();
        for i in 0..unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() } {
            let dev = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(i) };
            backend_devices.push(dev);
        }
        let mut devices = Vec::new();
        for &dev in self.devices.iter() {
            if dev.is_null() {
                break;
            }
            if let Some((index, _)) = backend_devices
                .iter()
                .enumerate()
                .find(|&(_i, &d)| d == dev)
            {
                devices.push(index);
            }
        }
        devices
    }

    /// sets the number of gpu layers to offload to the GPU.
    /// ```
    /// # use llama_cpp_bindings::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default();
    /// let params = params.with_n_gpu_layers(1);
    /// assert_eq!(params.n_gpu_layers(), 1);
    /// ```
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        // The only way this conversion can fail is if u32 overflows the i32 - in which case we set
        // to MAX
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    /// sets the main GPU
    ///
    /// To enable this option, you must set `split_mode` to `LlamaSplitMode::None` to enable single GPU mode.
    #[must_use]
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    /// sets `vocab_only`
    #[must_use]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    /// sets `use_mlock`
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    /// sets `split_mode`
    #[must_use]
    pub fn with_split_mode(mut self, split_mode: LlamaSplitMode) -> Self {
        self.params.split_mode = split_mode.into();
        self
    }

    /// sets `devices`
    ///
    /// The devices are specified as indices that correspond to the ggml backend device indices.
    ///
    /// The maximum number of devices is 16.
    ///
    /// You don't need to specify CPU or ACCEL devices.
    ///
    /// # Errors
    /// Returns `LlamaCppError::BackendDeviceNotFound` if any device index is invalid.
    pub fn with_devices(mut self, devices: &[usize]) -> Result<Self, LlamaCppError> {
        for dev in self.devices.iter_mut() {
            *dev = std::ptr::null_mut();
        }
        // Check device count
        let max_devices = crate::max_devices().min(LLAMA_CPP_MAX_DEVICES);
        if devices.len() > max_devices {
            return Err(LlamaCppError::MaxDevicesExceeded(max_devices));
        }
        for (i, &dev) in devices.iter().enumerate() {
            if dev >= unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() } {
                return Err(LlamaCppError::BackendDeviceNotFound(dev));
            }
            let backend_dev = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(dev) };
            self.devices[i] = backend_dev;
        }
        if self.devices.is_empty() {
            self.params.devices = std::ptr::null_mut();
        } else {
            self.params.devices = self.devices.as_mut_ptr();
        }
        Ok(self)
    }
}

/// Default parameters for `LlamaModel`. (as defined in llama.cpp by `llama_model_default_params`)
/// ```
/// # use llama_cpp_bindings::model::params::LlamaModelParams;
/// use llama_cpp_bindings::model::params::LlamaSplitMode;
/// let params = LlamaModelParams::default();
/// assert_eq!(params.n_gpu_layers(), -1, "n_gpu_layers should be -1");
/// assert_eq!(params.main_gpu(), 0, "main_gpu should be 0");
/// assert_eq!(params.vocab_only(), false, "vocab_only should be false");
/// assert_eq!(params.use_mmap(), true, "use_mmap should be true");
/// assert_eq!(params.use_mlock(), false, "use_mlock should be false");
/// assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer), "split_mode should be LAYER");
/// assert_eq!(params.devices().len(), 0, "devices should be empty");
/// ```
impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params = unsafe { llama_cpp_bindings_sys::llama_model_default_params() };
        LlamaModelParams {
            params: default_params,
            // push the next one to ensure we maintain the iterator invariant of ending with a 0
            kv_overrides: vec![llama_cpp_bindings_sys::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            }],
            buft_overrides: vec![llama_cpp_bindings_sys::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            }],
            devices: Box::pin([std::ptr::null_mut(); 16]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaModelParams, LlamaSplitMode, LlamaSplitModeParseError};

    #[test]
    fn split_mode_try_from_i32_none() {
        assert_eq!(LlamaSplitMode::try_from(0_i32), Ok(LlamaSplitMode::None));
    }

    #[test]
    fn split_mode_try_from_i32_layer() {
        assert_eq!(LlamaSplitMode::try_from(1_i32), Ok(LlamaSplitMode::Layer));
    }

    #[test]
    fn split_mode_try_from_i32_row() {
        assert_eq!(LlamaSplitMode::try_from(2_i32), Ok(LlamaSplitMode::Row));
    }

    #[test]
    fn split_mode_try_from_i32_invalid() {
        assert_eq!(
            LlamaSplitMode::try_from(99_i32),
            Err(LlamaSplitModeParseError(99))
        );
    }

    #[test]
    fn split_mode_try_from_u32_all_variants() {
        assert_eq!(LlamaSplitMode::try_from(0_u32), Ok(LlamaSplitMode::None));
        assert_eq!(LlamaSplitMode::try_from(1_u32), Ok(LlamaSplitMode::Layer));
        assert_eq!(LlamaSplitMode::try_from(2_u32), Ok(LlamaSplitMode::Row));
    }

    #[test]
    fn split_mode_try_from_u32_invalid() {
        assert!(LlamaSplitMode::try_from(99_u32).is_err());
    }

    #[test]
    fn split_mode_into_i32_roundtrip() {
        let variants = [
            (LlamaSplitMode::None, 0_i32),
            (LlamaSplitMode::Layer, 1_i32),
            (LlamaSplitMode::Row, 2_i32),
        ];

        for (variant, expected) in variants {
            let as_i32: i32 = variant.into();
            assert_eq!(as_i32, expected);
            assert_eq!(LlamaSplitMode::try_from(as_i32), Ok(variant));
        }
    }

    #[test]
    fn split_mode_into_u32_roundtrip() {
        let variants = [
            (LlamaSplitMode::None, 0_u32),
            (LlamaSplitMode::Layer, 1_u32),
            (LlamaSplitMode::Row, 2_u32),
        ];

        for (variant, expected) in variants {
            let as_u32: u32 = variant.into();
            assert_eq!(as_u32, expected);
            assert_eq!(LlamaSplitMode::try_from(as_u32), Ok(variant));
        }
    }

    #[test]
    fn split_mode_default_is_layer() {
        assert_eq!(LlamaSplitMode::default(), LlamaSplitMode::Layer);
    }

    #[test]
    fn default_params_have_expected_values() {
        let params = LlamaModelParams::default();

        assert_eq!(params.n_gpu_layers(), -1);
        assert_eq!(params.main_gpu(), 0);
        assert!(!params.vocab_only());
        assert!(params.use_mmap());
        assert!(!params.use_mlock());
        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer));
        assert!(params.devices().is_empty());
    }

    #[test]
    fn n_gpu_layers_roundtrip() {
        let params = LlamaModelParams::default().with_n_gpu_layers(32);

        assert_eq!(params.n_gpu_layers(), 32);
    }

    #[test]
    fn n_gpu_layers_overflow_clamps_to_max() {
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);

        assert_eq!(params.n_gpu_layers(), i32::MAX);
    }

    #[test]
    fn main_gpu_roundtrip() {
        let params = LlamaModelParams::default().with_main_gpu(2);

        assert_eq!(params.main_gpu(), 2);
    }

    #[test]
    fn vocab_only_roundtrip() {
        let params = LlamaModelParams::default().with_vocab_only(true);

        assert!(params.vocab_only());
    }

    #[test]
    fn use_mlock_roundtrip() {
        let params = LlamaModelParams::default().with_use_mlock(true);

        assert!(params.use_mlock());
    }

    #[test]
    fn split_mode_roundtrip() {
        let params = LlamaModelParams::default().with_split_mode(LlamaSplitMode::Row);

        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Row));
    }

    #[test]
    fn debug_format_includes_fields() {
        let params = LlamaModelParams::default();
        let debug_output = format!("{params:?}");

        assert!(debug_output.contains("n_gpu_layers"));
        assert!(debug_output.contains("main_gpu"));
        assert!(debug_output.contains("vocab_only"));
        assert!(debug_output.contains("use_mmap"));
        assert!(debug_output.contains("use_mlock"));
        assert!(debug_output.contains("split_mode"));
    }
}
