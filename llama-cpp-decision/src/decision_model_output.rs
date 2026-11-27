use llama_cpp_params::llama_context_params::LlamaContextParams;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionModelOutput {
    Logits,
    PooledEmbeddings,
    TokenEmbeddings,
}

impl DecisionModelOutput {
    /// `context_params` configured the way llama.cpp's `common_init_result` configures the
    /// context of a decision model that reads this output.
    #[must_use]
    pub fn context_params(self, context_params: LlamaContextParams) -> LlamaContextParams {
        match self {
            Self::Logits => context_params,
            Self::PooledEmbeddings => {
                Self::reading_embeddings(context_params, LlamaPoolingType::Unspecified)
            }
            Self::TokenEmbeddings => {
                Self::reading_embeddings(context_params, LlamaPoolingType::None)
            }
        }
    }

    fn reading_embeddings(
        context_params: LlamaContextParams,
        pooling_type: LlamaPoolingType,
    ) -> LlamaContextParams {
        let n_batch = context_params.n_batch();

        context_params
            .with_embeddings(true)
            .with_pooling_type(pooling_type)
            .with_n_outputs_max(n_batch)
            .with_n_outputs_max_per_seq(1)
            .with_n_batch(n_batch.min(context_params.n_ubatch()))
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_params::llama_context_params::LlamaContextParams;

    use super::DecisionModelOutput;

    #[test]
    fn a_decision_model_reading_embeddings_decodes_no_more_than_a_micro_batch_at_once() {
        assert_eq!(
            DecisionModelOutput::TokenEmbeddings
                .context_params(
                    LlamaContextParams::default()
                        .with_n_batch(2048)
                        .with_n_ubatch(512)
                )
                .n_batch(),
            512
        );
    }

    #[test]
    fn a_decision_model_reading_logits_keeps_its_batch() {
        assert_eq!(
            DecisionModelOutput::Logits
                .context_params(
                    LlamaContextParams::default()
                        .with_n_batch(2048)
                        .with_n_ubatch(512)
                )
                .n_batch(),
            2048
        );
    }
}
