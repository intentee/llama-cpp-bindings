#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KvCacheType {
    Unknown(llama_cpp_bindings_sys::ggml_type),
    F32,
    F16,
    Q40,
    Q41,
    Q50,
    Q51,
    Q80,
    Q81,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq10,
    Tq20,
    Mxfp4,
    Nvfp4,
    Q10,
    Q20,
}

impl From<KvCacheType> for llama_cpp_bindings_sys::ggml_type {
    fn from(value: KvCacheType) -> Self {
        match value {
            KvCacheType::Unknown(raw) => raw,
            KvCacheType::F32 => llama_cpp_bindings_sys::GGML_TYPE_F32,
            KvCacheType::F16 => llama_cpp_bindings_sys::GGML_TYPE_F16,
            KvCacheType::Q40 => llama_cpp_bindings_sys::GGML_TYPE_Q4_0,
            KvCacheType::Q41 => llama_cpp_bindings_sys::GGML_TYPE_Q4_1,
            KvCacheType::Q50 => llama_cpp_bindings_sys::GGML_TYPE_Q5_0,
            KvCacheType::Q51 => llama_cpp_bindings_sys::GGML_TYPE_Q5_1,
            KvCacheType::Q80 => llama_cpp_bindings_sys::GGML_TYPE_Q8_0,
            KvCacheType::Q81 => llama_cpp_bindings_sys::GGML_TYPE_Q8_1,
            KvCacheType::Q2K => llama_cpp_bindings_sys::GGML_TYPE_Q2_K,
            KvCacheType::Q3K => llama_cpp_bindings_sys::GGML_TYPE_Q3_K,
            KvCacheType::Q4K => llama_cpp_bindings_sys::GGML_TYPE_Q4_K,
            KvCacheType::Q5K => llama_cpp_bindings_sys::GGML_TYPE_Q5_K,
            KvCacheType::Q6K => llama_cpp_bindings_sys::GGML_TYPE_Q6_K,
            KvCacheType::Q8K => llama_cpp_bindings_sys::GGML_TYPE_Q8_K,
            KvCacheType::Iq2Xxs => llama_cpp_bindings_sys::GGML_TYPE_IQ2_XXS,
            KvCacheType::Iq2Xs => llama_cpp_bindings_sys::GGML_TYPE_IQ2_XS,
            KvCacheType::Iq3Xxs => llama_cpp_bindings_sys::GGML_TYPE_IQ3_XXS,
            KvCacheType::Iq1S => llama_cpp_bindings_sys::GGML_TYPE_IQ1_S,
            KvCacheType::Iq4Nl => llama_cpp_bindings_sys::GGML_TYPE_IQ4_NL,
            KvCacheType::Iq3S => llama_cpp_bindings_sys::GGML_TYPE_IQ3_S,
            KvCacheType::Iq2S => llama_cpp_bindings_sys::GGML_TYPE_IQ2_S,
            KvCacheType::Iq4Xs => llama_cpp_bindings_sys::GGML_TYPE_IQ4_XS,
            KvCacheType::I8 => llama_cpp_bindings_sys::GGML_TYPE_I8,
            KvCacheType::I16 => llama_cpp_bindings_sys::GGML_TYPE_I16,
            KvCacheType::I32 => llama_cpp_bindings_sys::GGML_TYPE_I32,
            KvCacheType::I64 => llama_cpp_bindings_sys::GGML_TYPE_I64,
            KvCacheType::F64 => llama_cpp_bindings_sys::GGML_TYPE_F64,
            KvCacheType::Iq1M => llama_cpp_bindings_sys::GGML_TYPE_IQ1_M,
            KvCacheType::Bf16 => llama_cpp_bindings_sys::GGML_TYPE_BF16,
            KvCacheType::Tq10 => llama_cpp_bindings_sys::GGML_TYPE_TQ1_0,
            KvCacheType::Tq20 => llama_cpp_bindings_sys::GGML_TYPE_TQ2_0,
            KvCacheType::Mxfp4 => llama_cpp_bindings_sys::GGML_TYPE_MXFP4,
            KvCacheType::Nvfp4 => llama_cpp_bindings_sys::GGML_TYPE_NVFP4,
            KvCacheType::Q10 => llama_cpp_bindings_sys::GGML_TYPE_Q1_0,
            KvCacheType::Q20 => llama_cpp_bindings_sys::GGML_TYPE_Q2_0,
        }
    }
}

impl From<llama_cpp_bindings_sys::ggml_type> for KvCacheType {
    fn from(value: llama_cpp_bindings_sys::ggml_type) -> Self {
        match value {
            llama_cpp_bindings_sys::GGML_TYPE_F32 => Self::F32,
            llama_cpp_bindings_sys::GGML_TYPE_F16 => Self::F16,
            llama_cpp_bindings_sys::GGML_TYPE_Q4_0 => Self::Q40,
            llama_cpp_bindings_sys::GGML_TYPE_Q4_1 => Self::Q41,
            llama_cpp_bindings_sys::GGML_TYPE_Q5_0 => Self::Q50,
            llama_cpp_bindings_sys::GGML_TYPE_Q5_1 => Self::Q51,
            llama_cpp_bindings_sys::GGML_TYPE_Q8_0 => Self::Q80,
            llama_cpp_bindings_sys::GGML_TYPE_Q8_1 => Self::Q81,
            llama_cpp_bindings_sys::GGML_TYPE_Q2_K => Self::Q2K,
            llama_cpp_bindings_sys::GGML_TYPE_Q3_K => Self::Q3K,
            llama_cpp_bindings_sys::GGML_TYPE_Q4_K => Self::Q4K,
            llama_cpp_bindings_sys::GGML_TYPE_Q5_K => Self::Q5K,
            llama_cpp_bindings_sys::GGML_TYPE_Q6_K => Self::Q6K,
            llama_cpp_bindings_sys::GGML_TYPE_Q8_K => Self::Q8K,
            llama_cpp_bindings_sys::GGML_TYPE_IQ2_XXS => Self::Iq2Xxs,
            llama_cpp_bindings_sys::GGML_TYPE_IQ2_XS => Self::Iq2Xs,
            llama_cpp_bindings_sys::GGML_TYPE_IQ3_XXS => Self::Iq3Xxs,
            llama_cpp_bindings_sys::GGML_TYPE_IQ1_S => Self::Iq1S,
            llama_cpp_bindings_sys::GGML_TYPE_IQ4_NL => Self::Iq4Nl,
            llama_cpp_bindings_sys::GGML_TYPE_IQ3_S => Self::Iq3S,
            llama_cpp_bindings_sys::GGML_TYPE_IQ2_S => Self::Iq2S,
            llama_cpp_bindings_sys::GGML_TYPE_IQ4_XS => Self::Iq4Xs,
            llama_cpp_bindings_sys::GGML_TYPE_I8 => Self::I8,
            llama_cpp_bindings_sys::GGML_TYPE_I16 => Self::I16,
            llama_cpp_bindings_sys::GGML_TYPE_I32 => Self::I32,
            llama_cpp_bindings_sys::GGML_TYPE_I64 => Self::I64,
            llama_cpp_bindings_sys::GGML_TYPE_F64 => Self::F64,
            llama_cpp_bindings_sys::GGML_TYPE_IQ1_M => Self::Iq1M,
            llama_cpp_bindings_sys::GGML_TYPE_BF16 => Self::Bf16,
            llama_cpp_bindings_sys::GGML_TYPE_TQ1_0 => Self::Tq10,
            llama_cpp_bindings_sys::GGML_TYPE_TQ2_0 => Self::Tq20,
            llama_cpp_bindings_sys::GGML_TYPE_MXFP4 => Self::Mxfp4,
            llama_cpp_bindings_sys::GGML_TYPE_NVFP4 => Self::Nvfp4,
            llama_cpp_bindings_sys::GGML_TYPE_Q1_0 => Self::Q10,
            llama_cpp_bindings_sys::GGML_TYPE_Q2_0 => Self::Q20,
            _ => Self::Unknown(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::KvCacheType;

    fn is_retired_upstream_slot(raw: llama_cpp_bindings_sys::ggml_type) -> bool {
        unsafe { llama_cpp_bindings_sys::ggml_blck_size(raw) == 0 }
    }

    #[test]
    fn every_upstream_ggml_type_maps_to_a_known_variant() {
        for raw in 0..llama_cpp_bindings_sys::GGML_TYPE_COUNT {
            if is_retired_upstream_slot(raw) {
                continue;
            }

            let upstream_name =
                unsafe { CStr::from_ptr(llama_cpp_bindings_sys::ggml_type_name(raw)) }
                    .to_string_lossy();

            assert_ne!(
                KvCacheType::from(raw),
                KvCacheType::Unknown(raw),
                "ggml type {raw} (\"{upstream_name}\") has no matching KvCacheType variant, so it \
                 silently degrades to Unknown; add the variant to keep the mirror complete"
            );
        }
    }

    #[test]
    fn kv_cache_type_unknown_preserves_raw_value() {
        let unknown_raw: llama_cpp_bindings_sys::ggml_type = 99999;
        let cache_type = KvCacheType::from(unknown_raw);

        assert_eq!(cache_type, KvCacheType::Unknown(99999));

        let back: llama_cpp_bindings_sys::ggml_type = cache_type.into();

        assert_eq!(back, 99999);
    }

    #[test]
    fn every_kv_cache_type_converts_to_its_ggml_type_and_back() {
        let all_variants = [
            KvCacheType::F32,
            KvCacheType::F16,
            KvCacheType::Q40,
            KvCacheType::Q41,
            KvCacheType::Q50,
            KvCacheType::Q51,
            KvCacheType::Q80,
            KvCacheType::Q81,
            KvCacheType::Q2K,
            KvCacheType::Q3K,
            KvCacheType::Q4K,
            KvCacheType::Q5K,
            KvCacheType::Q6K,
            KvCacheType::Q8K,
            KvCacheType::Iq2Xxs,
            KvCacheType::Iq2Xs,
            KvCacheType::Iq3Xxs,
            KvCacheType::Iq1S,
            KvCacheType::Iq4Nl,
            KvCacheType::Iq3S,
            KvCacheType::Iq2S,
            KvCacheType::Iq4Xs,
            KvCacheType::I8,
            KvCacheType::I16,
            KvCacheType::I32,
            KvCacheType::I64,
            KvCacheType::F64,
            KvCacheType::Iq1M,
            KvCacheType::Bf16,
            KvCacheType::Tq10,
            KvCacheType::Tq20,
            KvCacheType::Mxfp4,
            KvCacheType::Nvfp4,
            KvCacheType::Q10,
            KvCacheType::Q20,
        ];

        for variant in all_variants {
            let ggml_type: llama_cpp_bindings_sys::ggml_type = variant.into();
            let back = KvCacheType::from(ggml_type);

            assert_eq!(back, variant);
        }
    }
}
