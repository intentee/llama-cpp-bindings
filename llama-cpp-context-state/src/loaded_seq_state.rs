use llama_cpp_model::llama_token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedSeqState {
    pub tokens: Vec<LlamaToken>,
    pub bytes_read: usize,
}
