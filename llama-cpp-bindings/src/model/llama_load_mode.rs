use crate::model::llama_load_mode_parse_error::LlamaLoadModeParseError;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LlamaLoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl TryFrom<llama_cpp_bindings_sys::llama_load_mode> for LlamaLoadMode {
    type Error = LlamaLoadModeParseError;

    fn try_from(value: llama_cpp_bindings_sys::llama_load_mode) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_AUTO => Ok(Self::Auto),
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MMAP => Ok(Self::Mmap),
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MLOCK => Ok(Self::Mlock),
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MMAP_MLOCK => Ok(Self::MmapMlock),
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_DIRECT_IO => Ok(Self::DirectIo),
            value => Err(LlamaLoadModeParseError { value }),
        }
    }
}

impl From<LlamaLoadMode> for llama_cpp_bindings_sys::llama_load_mode {
    fn from(value: LlamaLoadMode) -> Self {
        match value {
            LlamaLoadMode::Auto => llama_cpp_bindings_sys::LLAMA_LOAD_MODE_AUTO,
            LlamaLoadMode::None => llama_cpp_bindings_sys::LLAMA_LOAD_MODE_NONE,
            LlamaLoadMode::Mmap => llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MMAP,
            LlamaLoadMode::Mlock => llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MLOCK,
            LlamaLoadMode::MmapMlock => llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MMAP_MLOCK,
            LlamaLoadMode::DirectIo => llama_cpp_bindings_sys::LLAMA_LOAD_MODE_DIRECT_IO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaLoadMode;

    const LOAD_MODES: &[(LlamaLoadMode, llama_cpp_bindings_sys::llama_load_mode)] = &[
        (
            LlamaLoadMode::Auto,
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_AUTO,
        ),
        (
            LlamaLoadMode::None,
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_NONE,
        ),
        (
            LlamaLoadMode::Mmap,
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MMAP,
        ),
        (
            LlamaLoadMode::Mlock,
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MLOCK,
        ),
        (
            LlamaLoadMode::MmapMlock,
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_MMAP_MLOCK,
        ),
        (
            LlamaLoadMode::DirectIo,
            llama_cpp_bindings_sys::LLAMA_LOAD_MODE_DIRECT_IO,
        ),
    ];

    #[test]
    fn every_rust_load_mode_maps_to_its_ffi_value() {
        for &(load_mode, ffi_value) in LOAD_MODES {
            assert_eq!(
                llama_cpp_bindings_sys::llama_load_mode::from(load_mode),
                ffi_value
            );
        }
    }

    #[test]
    fn every_ffi_load_mode_maps_to_its_rust_value() {
        for &(load_mode, ffi_value) in LOAD_MODES {
            assert_eq!(LlamaLoadMode::try_from(ffi_value), Ok(load_mode));
        }
    }

    #[test]
    fn unknown_ffi_load_mode_preserves_its_value() {
        assert_eq!(
            LlamaLoadMode::try_from(i32::MAX),
            Err(super::LlamaLoadModeParseError { value: i32::MAX })
        );
    }
}
