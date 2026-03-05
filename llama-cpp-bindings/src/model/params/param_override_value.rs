/// An override value for a model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamOverrideValue {
    /// A boolean value
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
}
