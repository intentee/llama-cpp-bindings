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

const LLAMA_SPLIT_MODE_NONE: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_NONE as i8;
const LLAMA_SPLIT_MODE_LAYER: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_LAYER as i8;
const LLAMA_SPLIT_MODE_ROW: i8 = llama_cpp_bindings_sys::LLAMA_SPLIT_MODE_ROW as i8;

/// An error that occurs when unknown split mode is encountered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSplitModeParseError {
    /// The value that could not be parsed as a split mode.
    pub value: i32,
    /// Additional context about why the parse failed.
    pub context: String,
}

/// Create a `LlamaSplitMode` from a `i32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<i32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|convert_error| LlamaSplitModeParseError {
                value,
                context: format!("i32 to i8 conversion failed: {convert_error}"),
            })?;

        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            _ => Err(LlamaSplitModeParseError {
                value,
                context: format!("unknown split mode value: {value}"),
            }),
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
        let clamped_value = i32::try_from(value).unwrap_or(i32::MAX);
        let i8_value = value
            .try_into()
            .map_err(|convert_error| LlamaSplitModeParseError {
                value: clamped_value,
                context: format!("u32 to i8 conversion failed: {convert_error}"),
            })?;

        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            _ => Err(LlamaSplitModeParseError {
                value: clamped_value,
                context: format!("unknown split mode value: {value}"),
            }),
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

#[cfg(test)]
mod tests {
    use super::{LlamaSplitMode, LlamaSplitModeParseError};

    #[test]
    fn try_from_i32_invalid() {
        let result = LlamaSplitMode::try_from(99_i32);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error.value, 99);
    }

    #[test]
    fn try_from_u32_invalid() {
        assert!(LlamaSplitMode::try_from(99_u32).is_err());
    }
}
