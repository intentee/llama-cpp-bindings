use crate::model::llama_lazy_mode_parse_error::LlamaLazyModeParseError;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LlamaLazyMode {
    Off,
    #[default]
    Auto,
    On,
}

impl TryFrom<llama_cpp_bindings_sys::llama_lazy_mode> for LlamaLazyMode {
    type Error = LlamaLazyModeParseError;

    fn try_from(value: llama_cpp_bindings_sys::llama_lazy_mode) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_LAZY_MODE_OFF => Ok(Self::Off),
            llama_cpp_bindings_sys::LLAMA_LAZY_MODE_AUTO => Ok(Self::Auto),
            llama_cpp_bindings_sys::LLAMA_LAZY_MODE_ON => Ok(Self::On),
            value => Err(LlamaLazyModeParseError {
                value: i64::from(value),
            }),
        }
    }
}

impl From<LlamaLazyMode> for llama_cpp_bindings_sys::llama_lazy_mode {
    fn from(value: LlamaLazyMode) -> Self {
        match value {
            LlamaLazyMode::Off => llama_cpp_bindings_sys::LLAMA_LAZY_MODE_OFF,
            LlamaLazyMode::Auto => llama_cpp_bindings_sys::LLAMA_LAZY_MODE_AUTO,
            LlamaLazyMode::On => llama_cpp_bindings_sys::LLAMA_LAZY_MODE_ON,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaLazyMode;
    use crate::model::llama_lazy_mode_parse_error::LlamaLazyModeParseError;

    const LAZY_MODES: &[(LlamaLazyMode, llama_cpp_bindings_sys::llama_lazy_mode)] = &[
        (
            LlamaLazyMode::Off,
            llama_cpp_bindings_sys::LLAMA_LAZY_MODE_OFF,
        ),
        (
            LlamaLazyMode::Auto,
            llama_cpp_bindings_sys::LLAMA_LAZY_MODE_AUTO,
        ),
        (
            LlamaLazyMode::On,
            llama_cpp_bindings_sys::LLAMA_LAZY_MODE_ON,
        ),
    ];

    #[test]
    fn every_rust_lazy_mode_maps_to_its_ffi_value() {
        for &(lazy_mode, ffi_value) in LAZY_MODES {
            assert_eq!(
                llama_cpp_bindings_sys::llama_lazy_mode::from(lazy_mode),
                ffi_value
            );
        }
    }

    #[test]
    fn every_ffi_lazy_mode_maps_to_its_rust_value() {
        for &(lazy_mode, ffi_value) in LAZY_MODES {
            assert_eq!(LlamaLazyMode::try_from(ffi_value), Ok(lazy_mode));
        }
    }

    #[test]
    fn unknown_ffi_lazy_mode_preserves_its_value() {
        let unknown = llama_cpp_bindings_sys::llama_lazy_mode::MAX;

        assert_eq!(
            LlamaLazyMode::try_from(unknown),
            Err(LlamaLazyModeParseError {
                value: i64::from(unknown)
            })
        );
    }

    #[test]
    fn default_lazy_mode_is_auto() {
        assert_eq!(LlamaLazyMode::default(), LlamaLazyMode::Auto);
    }
}
