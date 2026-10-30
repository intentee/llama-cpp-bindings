use llama_cpp_bindings_sys::{llama_pos, llama_seq_id};

use crate::token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaBatchEntry {
    pub token: LlamaToken,
    pub pos: llama_pos,
    pub seq_ids: Vec<llama_seq_id>,
    pub output: bool,
}
