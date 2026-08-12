use crate::model::llama_load_mode_parse_error::LlamaLoadModeParseError;

/// How llama.cpp should bring model weights into memory.
///
/// [`LlamaLoadMode::Auto`] is upstream's default: it memory-maps the weights
/// unless a participating device reports no mmap support, in which case
/// llama.cpp falls back to a plain read.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LlamaLoadMode {
    #[default]
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl LlamaLoadMode {
    /// The same mode with memory mapping removed, keeping any mlock request.
    ///
    /// Upstream models the mode as one value rather than a bit set, so
    /// dropping mmap is a remap rather than a bit clear.
    #[must_use]
    pub const fn without_mmap(self) -> Self {
        match self {
            Self::Auto | Self::None | Self::Mmap => Self::None,
            Self::Mlock | Self::MmapMlock => Self::Mlock,
            Self::DirectIo => Self::DirectIo,
        }
    }
}

/// # Errors
/// Returns `LlamaLoadModeParseError` if the value does not correspond to a valid `LlamaLoadMode`.
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
            _ => Err(LlamaLoadModeParseError {
                value,
                context: format!("unknown load mode value: {value}"),
            }),
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

    #[test]
    fn try_from_invalid_reports_the_value() {
        let result = LlamaLoadMode::try_from(99);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().value, 99);
    }

    #[test]
    fn every_variant_roundtrips_through_the_upstream_value() {
        for mode in [
            LlamaLoadMode::Auto,
            LlamaLoadMode::None,
            LlamaLoadMode::Mmap,
            LlamaLoadMode::Mlock,
            LlamaLoadMode::MmapMlock,
            LlamaLoadMode::DirectIo,
        ] {
            let raw = llama_cpp_bindings_sys::llama_load_mode::from(mode);

            assert_eq!(LlamaLoadMode::try_from(raw), Ok(mode));
        }
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(LlamaLoadMode::default(), LlamaLoadMode::Auto);
    }

    #[test]
    fn without_mmap_drops_mapping_and_keeps_mlock() {
        assert_eq!(LlamaLoadMode::Auto.without_mmap(), LlamaLoadMode::None);
        assert_eq!(LlamaLoadMode::None.without_mmap(), LlamaLoadMode::None);
        assert_eq!(LlamaLoadMode::Mmap.without_mmap(), LlamaLoadMode::None);
        assert_eq!(LlamaLoadMode::Mlock.without_mmap(), LlamaLoadMode::Mlock);
        assert_eq!(
            LlamaLoadMode::MmapMlock.without_mmap(),
            LlamaLoadMode::Mlock
        );
        assert_eq!(
            LlamaLoadMode::DirectIo.without_mmap(),
            LlamaLoadMode::DirectIo
        );
    }
}
