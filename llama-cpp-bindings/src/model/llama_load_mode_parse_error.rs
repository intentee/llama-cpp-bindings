#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("unknown llama load mode {value}")]
pub struct LlamaLoadModeParseError {
    pub value: i32,
}
