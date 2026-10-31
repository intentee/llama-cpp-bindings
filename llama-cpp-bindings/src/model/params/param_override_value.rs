use std::ffi::c_char;

use crate::model::params::unknown_kv_override_tag::UnknownKvOverrideTag;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamOverrideValue {
    Bool(bool),
    Float(f64),
    Int(i64),
    Str([c_char; 128]),
}

impl ParamOverrideValue {
    #[must_use]
    pub const fn tag(&self) -> llama_cpp_bindings_sys::llama_model_kv_override_type {
        match self {
            Self::Bool(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            Self::Float(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            Self::Int(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            Self::Str(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
        }
    }

    #[must_use]
    pub const fn value(&self) -> llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
        match self {
            Self::Bool(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_bool: *value }
            }
            Self::Float(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_f64: *value }
            }
            Self::Int(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_i64: *value }
            }
            Self::Str(c_string) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_str: *c_string }
            }
        }
    }
}

impl TryFrom<&llama_cpp_bindings_sys::llama_model_kv_override> for ParamOverrideValue {
    type Error = UnknownKvOverrideTag;

    fn try_from(
        llama_cpp_bindings_sys::llama_model_kv_override {
            key: _,
            tag,
            __bindgen_anon_1,
        }: &llama_cpp_bindings_sys::llama_model_kv_override,
    ) -> Result<Self, Self::Error> {
        match *tag {
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT => {
                Ok(Self::Int(unsafe { __bindgen_anon_1.val_i64 }))
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT => {
                Ok(Self::Float(unsafe { __bindgen_anon_1.val_f64 }))
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL => {
                Ok(Self::Bool(unsafe { __bindgen_anon_1.val_bool }))
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR => {
                Ok(Self::Str(unsafe { __bindgen_anon_1.val_str }))
            }
            unknown_tag => Err(UnknownKvOverrideTag(unknown_tag)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use super::ParamOverrideValue;

    #[test]
    fn each_override_value_is_tagged_with_its_llama_cpp_type() {
        assert_eq!(
            [
                ParamOverrideValue::Bool(true),
                ParamOverrideValue::Float(1.23),
                ParamOverrideValue::Int(42),
                ParamOverrideValue::Str([0; 128]),
            ]
            .map(|value| value.tag()),
            [
                llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
                llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
                llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
                llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
            ]
        );
    }

    #[test]
    fn a_bool_override_stores_its_value_in_the_bool_field() {
        let value = ParamOverrideValue::Bool(true);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_bool };

        assert!(result);
    }

    #[test]
    fn a_float_override_stores_its_value_in_the_f64_field() {
        let value = ParamOverrideValue::Float(1.23);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_f64 };

        assert!((result - 1.23).abs() < f64::EPSILON);
    }

    #[test]
    fn an_int_override_stores_its_value_in_the_i64_field() {
        let value = ParamOverrideValue::Int(99);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_i64 };

        assert_eq!(result, 99);
    }

    #[test]
    fn an_int_kv_override_reads_as_an_int_value() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_i64: 123,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Int(123));
    }

    #[test]
    fn a_float_kv_override_reads_as_a_float_value() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_f64: 1.5,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Float(1.5));
    }

    #[test]
    fn a_bool_kv_override_reads_as_a_bool_value() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_bool: false,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Bool(false));
    }

    #[test]
    fn a_str_override_stores_its_bytes_in_the_str_field() {
        let mut str_data = [0i8; 128];
        str_data[0] = b'h'.cast_signed();
        str_data[1] = b'i'.cast_signed();

        let value = ParamOverrideValue::Str(str_data);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_str };

        assert_eq!(result[0], b'h'.cast_signed());
        assert_eq!(result[1], b'i'.cast_signed());
    }

    #[test]
    fn a_str_kv_override_reads_as_a_str_value() {
        let mut str_data = [0i8; 128];
        str_data[0] = b'a'.cast_signed();
        str_data[1] = b'b'.cast_signed();

        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_str: str_data,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Str(str_data));
    }

    #[test]
    fn unknown_tag_returns_error() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: 9999,
            __bindgen_anon_1: unsafe { mem::zeroed() },
        };

        let result = ParamOverrideValue::try_from(&ffi_override);

        assert!(result.is_err());
    }
}
