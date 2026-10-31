use crate::model::llama_split_mode_parse_error::LlamaSplitModeParseError;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LlamaSplitMode {
    None,
    #[default]
    Layer,
    Row,
    Tensor,
}

/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<llama_cpp_bindings_sys::llama_split_mode> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: llama_cpp_bindings_sys::llama_split_mode) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError {
                value,
                context: format!("unknown split mode value: {value}"),
            }),
        }
    }
}

impl From<LlamaSplitMode> for llama_cpp_bindings_sys::llama_split_mode {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_NONE,
            LlamaSplitMode::Layer => llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_LAYER,
            LlamaSplitMode::Row => llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_ROW,
            LlamaSplitMode::Tensor => llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_TENSOR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaSplitMode;

    #[test]
    fn try_from_invalid_reports_the_value() {
        let result = LlamaSplitMode::try_from(99);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().value, 99);
    }

    #[test]
    fn every_llama_cpp_split_mode_converts_to_its_variant_and_back() {
        let llama_cpp_split_modes = [
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_NONE,
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_LAYER,
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_ROW,
            llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_TENSOR,
        ];
        let modes = llama_cpp_split_modes.map(LlamaSplitMode::try_from);

        assert_eq!(
            modes,
            [
                Ok(LlamaSplitMode::None),
                Ok(LlamaSplitMode::Layer),
                Ok(LlamaSplitMode::Row),
                Ok(LlamaSplitMode::Tensor),
            ]
        );
        assert_eq!(
            modes.map(|mode| mode.map(llama_cpp_bindings_sys::llama_split_mode::from)),
            llama_cpp_split_modes.map(Ok)
        );
    }

    #[test]
    fn default_is_layer() {
        assert_eq!(LlamaSplitMode::default(), LlamaSplitMode::Layer);
    }
}
