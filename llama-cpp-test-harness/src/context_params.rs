use std::num::NonZeroU32;

use llama_cpp_bindings::context::params::LlamaContextParams;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads_batch: Option<i32>,
    pub embeddings: bool,
}

impl ContextParams {
    #[must_use]
    pub fn into_llama_context_params(self) -> LlamaContextParams {
        let Self {
            n_ctx,
            n_batch,
            n_ubatch,
            n_seq_max,
            n_threads_batch,
            embeddings,
        } = self;
        let mut params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_ubatch)
            .with_n_seq_max(n_seq_max)
            .with_embeddings(embeddings);
        if let Some(threads) = n_threads_batch {
            params = params.with_n_threads_batch(threads);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use llama_cpp_bindings::context::params::LlamaContextParams;

    use super::ContextParams;

    const BASELINE: ContextParams = ContextParams {
        n_ctx: 512,
        n_batch: 128,
        n_ubatch: 64,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    #[test]
    fn into_llama_context_params_carries_all_fields() {
        let params = ContextParams {
            n_ctx: 1024,
            n_batch: 256,
            n_ubatch: 128,
            n_seq_max: 4,
            n_threads_batch: Some(8),
            embeddings: true,
        }
        .into_llama_context_params();

        assert_eq!(params.n_ctx(), NonZeroU32::new(1024));
        assert_eq!(params.n_batch(), 256);
        assert_eq!(params.n_ubatch(), 128);
        assert_eq!(params.n_seq_max(), 4);
        assert_eq!(params.n_threads_batch(), 8);
        assert!(params.embeddings());
    }

    #[test]
    fn into_llama_context_params_leaves_n_threads_batch_default_when_none() {
        let default_params = LlamaContextParams::default();
        let params = ContextParams {
            n_threads_batch: None,
            ..BASELINE
        }
        .into_llama_context_params();

        assert_eq!(
            params.context_params.n_threads_batch,
            default_params.context_params.n_threads_batch
        );
    }

    #[test]
    fn zero_n_ctx_means_model_default() {
        let params = ContextParams {
            n_ctx: 0,
            ..BASELINE
        }
        .into_llama_context_params();

        assert_eq!(params.n_ctx(), None);
    }

    #[test]
    fn differing_n_ctx_compares_unequal() {
        let one = ContextParams {
            n_ctx: 512,
            ..BASELINE
        };
        let two = ContextParams {
            n_ctx: 1024,
            ..BASELINE
        };

        assert_ne!(one, two);
    }

    #[test]
    fn differing_embeddings_compares_unequal() {
        let off = ContextParams {
            embeddings: false,
            ..BASELINE
        };
        let on = ContextParams {
            embeddings: true,
            ..BASELINE
        };

        assert_ne!(off, on);
    }

    #[test]
    fn differing_n_seq_max_compares_unequal() {
        let one = ContextParams {
            n_seq_max: 1,
            ..BASELINE
        };
        let two = ContextParams {
            n_seq_max: 4,
            ..BASELINE
        };

        assert_ne!(one, two);
    }

    #[test]
    fn differing_n_threads_batch_compares_unequal() {
        let none = ContextParams {
            n_threads_batch: None,
            ..BASELINE
        };
        let some = ContextParams {
            n_threads_batch: Some(8),
            ..BASELINE
        };

        assert_ne!(none, some);
    }
}
