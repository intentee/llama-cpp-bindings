#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("unknown llama lazy mode {value}")]
pub struct LlamaLazyModeParseError {
    pub value: i64,
}
