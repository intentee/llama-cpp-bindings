use crate::token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedSeqState {
    pub tokens: Vec<LlamaToken>,
    pub bytes_read: usize,
}
