//! Key-value overrides for a model.

use crate::model::params::LlamaModelParams;
use std::ffi::{CStr, CString};
use std::fmt::Debug;

/// An override value for a model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamOverrideValue {
    /// A string value
    Bool(bool),
    /// A float value
    Float(f64),
    /// A integer value
    Int(i64),
    /// A string value
    Str([std::os::raw::c_char; 128]),
}

impl ParamOverrideValue {
    /// Returns the FFI tag corresponding to this override value variant.
    #[must_use]
    pub fn tag(&self) -> llama_cpp_bindings_sys::llama_model_kv_override_type {
        match self {
            ParamOverrideValue::Bool(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            ParamOverrideValue::Float(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            ParamOverrideValue::Int(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            ParamOverrideValue::Str(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
        }
    }

    /// Returns the FFI union value for this override.
    #[must_use]
    pub fn value(&self) -> llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
        match self {
            ParamOverrideValue::Bool(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_bool: *value }
            }
            ParamOverrideValue::Float(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_f64: *value }
            }
            ParamOverrideValue::Int(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_i64: *value }
            }
            ParamOverrideValue::Str(c_string) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_str: *c_string }
            }
        }
    }
}

impl From<&llama_cpp_bindings_sys::llama_model_kv_override> for ParamOverrideValue {
    fn from(
        llama_cpp_bindings_sys::llama_model_kv_override {
            key: _,
            tag,
            __bindgen_anon_1,
        }: &llama_cpp_bindings_sys::llama_model_kv_override,
    ) -> Self {
        match *tag {
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT => {
                ParamOverrideValue::Int(unsafe { __bindgen_anon_1.val_i64 })
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT => {
                ParamOverrideValue::Float(unsafe { __bindgen_anon_1.val_f64 })
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL => {
                ParamOverrideValue::Bool(unsafe { __bindgen_anon_1.val_bool })
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR => {
                ParamOverrideValue::Str(unsafe { __bindgen_anon_1.val_str })
            }
            _ => unreachable!("Unknown tag of {tag}"),
        }
    }
}

/// A struct implementing [`IntoIterator`] over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrides<'model_params> {
    model_params: &'model_params LlamaModelParams,
}

impl KvOverrides<'_> {
    /// Creates a new `KvOverrides` view over the given model parameters.
    #[must_use]
    pub fn new(model_params: &LlamaModelParams) -> KvOverrides<'_> {
        KvOverrides { model_params }
    }
}

impl<'model_params> IntoIterator for KvOverrides<'model_params> {
    // I'm fairly certain this could be written returning by reference, but I'm not sure how to do it safely. I do not
    // expect this to be a performance bottleneck so the copy should be fine. (let me know if it's not fine!)
    type Item = (CString, ParamOverrideValue);
    type IntoIter = KvOverrideValueIterator<'model_params>;

    fn into_iter(self) -> Self::IntoIter {
        KvOverrideValueIterator {
            model_params: self.model_params,
            current: 0,
        }
    }
}

/// An iterator over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrideValueIterator<'model_params> {
    model_params: &'model_params LlamaModelParams,
    current: usize,
}

impl Iterator for KvOverrideValueIterator<'_> {
    type Item = (CString, ParamOverrideValue);

    fn next(&mut self) -> Option<Self::Item> {
        let overrides = self.model_params.params.kv_overrides;
        if overrides.is_null() {
            return None;
        }

        // SAFETY: llama.cpp seems to guarantee that the last element contains an empty key or is valid. We've checked
        // the prev one in the last iteration, the next one should be valid or 0 (and thus safe to deref)
        let current = unsafe { *overrides.add(self.current) };

        if current.key[0] == 0 {
            return None;
        }

        let value = ParamOverrideValue::from(&current);

        let key = unsafe { CStr::from_ptr(current.key.as_ptr()).to_owned() };

        self.current += 1;
        Some((key, value))
    }
}

#[cfg(test)]
mod tests {
    use super::ParamOverrideValue;

    #[test]
    fn tag_bool() {
        let value = ParamOverrideValue::Bool(true);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL
        );
    }

    #[test]
    fn tag_float() {
        let value = ParamOverrideValue::Float(3.14);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT
        );
    }

    #[test]
    fn tag_int() {
        let value = ParamOverrideValue::Int(42);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT
        );
    }

    #[test]
    fn tag_str() {
        let value = ParamOverrideValue::Str([0; 128]);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR
        );
    }

    #[test]
    fn value_bool_roundtrip() {
        let value = ParamOverrideValue::Bool(true);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_bool };

        assert!(result);
    }

    #[test]
    fn value_float_roundtrip() {
        let value = ParamOverrideValue::Float(2.718);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_f64 };

        assert!((result - 2.718).abs() < f64::EPSILON);
    }

    #[test]
    fn value_int_roundtrip() {
        let value = ParamOverrideValue::Int(99);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_i64 };

        assert_eq!(result, 99);
    }

    #[test]
    fn from_ffi_override_int() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_i64: 123,
            },
        };

        let value = ParamOverrideValue::from(&ffi_override);

        assert_eq!(value, ParamOverrideValue::Int(123));
    }

    #[test]
    fn from_ffi_override_float() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_f64: 1.5,
            },
        };

        let value = ParamOverrideValue::from(&ffi_override);

        assert_eq!(value, ParamOverrideValue::Float(1.5));
    }

    #[test]
    fn from_ffi_override_bool() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_bool: false,
            },
        };

        let value = ParamOverrideValue::from(&ffi_override);

        assert_eq!(value, ParamOverrideValue::Bool(false));
    }

    #[test]
    fn kv_overrides_empty_by_default() {
        let params = crate::model::params::LlamaModelParams::default();
        let overrides = params.kv_overrides();
        let count = overrides.into_iter().count();

        assert_eq!(count, 0);
    }
}
