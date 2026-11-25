use llama_cpp_params::llama_pooling_type::LlamaPoolingType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingPoolingChoice {
    Explicit(LlamaPoolingType),
    ModelDefault,
}
