use std::num::NonZeroU32;

use crate::llama_attention_type::LlamaAttentionType;
use crate::llama_context_params::LlamaContextParams;
use crate::model_output_traits::ModelOutputTraits;
use crate::params_error::ParamsError;
use crate::sequence_output::SequenceOutput;

pub struct SequenceOutputCapacity<'params> {
    pub context_params: &'params LlamaContextParams,
    pub model_output_traits: ModelOutputTraits,
}

impl SequenceOutputCapacity<'_> {
    /// Whether every sequence of a context llama.cpp creates from these params gets an output
    /// in one batch, the way `llama_context` sizes its output buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ParamsError::UnknownAttentionType`] when the context params hold an attention
    /// type llama.cpp does not define.
    pub fn sequence_output(&self) -> Result<SequenceOutput, ParamsError> {
        let Self {
            context_params,
            model_output_traits:
                ModelOutputTraits {
                    causal_attention,
                    has_encoder,
                    n_ctx_train,
                },
        } = self;
        let n_ctx = context_params.n_ctx().map_or(*n_ctx_train, NonZeroU32::get);
        let causal = match context_params.attention_type()? {
            LlamaAttentionType::Unspecified => *causal_attention,
            LlamaAttentionType::Causal => true,
            LlamaAttentionType::NonCausal => false,
        };
        let n_batch = if causal {
            n_ctx.min(context_params.n_batch())
        } else {
            context_params.n_batch()
        };
        let output_capacity = if context_params.n_outputs_max() == 0 || *has_encoder {
            n_batch
        } else {
            context_params.n_outputs_max()
        };
        let n_seq_max = context_params.n_seq_max().max(1);

        if n_seq_max > output_capacity {
            return Ok(SequenceOutput::ExceedsCapacity {
                n_seq_max,
                output_capacity,
            });
        }

        Ok(SequenceOutput::FitsCapacity)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::SequenceOutputCapacity;
    use crate::llama_attention_type::LlamaAttentionType;
    use crate::llama_context_params::LlamaContextParams;
    use crate::model_output_traits::ModelOutputTraits;
    use crate::sequence_output::SequenceOutput;

    const TWO_SEQUENCES_PAST_ONE_OUTPUT: SequenceOutput = SequenceOutput::ExceedsCapacity {
        n_seq_max: 2,
        output_capacity: 1,
    };

    const A_NON_CAUSAL_MODEL: ModelOutputTraits = ModelOutputTraits {
        causal_attention: false,
        has_encoder: false,
        n_ctx_train: 4096,
    };

    fn two_sequences() -> LlamaContextParams {
        LlamaContextParams::default()
            .with_attention_type(LlamaAttentionType::NonCausal)
            .with_n_batch(2)
            .with_n_ctx(NonZeroU32::new(4096))
            .with_n_seq_max(2)
    }

    fn sequence_output_of(
        context_params: &LlamaContextParams,
        model_output_traits: ModelOutputTraits,
    ) -> SequenceOutput {
        SequenceOutputCapacity {
            context_params,
            model_output_traits,
        }
        .sequence_output()
        .unwrap()
    }

    #[test]
    fn the_default_output_capacity_is_one_batch() {
        assert_eq!(
            sequence_output_of(&two_sequences().with_n_batch(1), A_NON_CAUSAL_MODEL),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn an_explicit_output_capacity_admits_more_sequences_than_one_batch() {
        assert_eq!(
            sequence_output_of(
                &two_sequences().with_n_batch(1).with_n_outputs_max(2),
                A_NON_CAUSAL_MODEL
            ),
            SequenceOutput::FitsCapacity
        );
    }

    #[test]
    fn an_encoder_model_outputs_at_most_one_batch() {
        assert_eq!(
            sequence_output_of(
                &two_sequences().with_n_batch(1).with_n_outputs_max(2),
                ModelOutputTraits {
                    has_encoder: true,
                    ..A_NON_CAUSAL_MODEL
                }
            ),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn a_causal_context_outputs_at_most_its_context_size() {
        assert_eq!(
            sequence_output_of(
                &two_sequences()
                    .with_attention_type(LlamaAttentionType::Causal)
                    .with_n_ctx(NonZeroU32::new(1)),
                A_NON_CAUSAL_MODEL
            ),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn a_non_causal_context_outputs_a_whole_batch_whatever_its_context_size() {
        assert_eq!(
            sequence_output_of(
                &two_sequences().with_n_ctx(NonZeroU32::new(1)),
                ModelOutputTraits {
                    causal_attention: true,
                    ..A_NON_CAUSAL_MODEL
                }
            ),
            SequenceOutput::FitsCapacity
        );
    }

    #[test]
    fn an_unspecified_attention_type_follows_the_model() {
        assert_eq!(
            sequence_output_of(
                &two_sequences()
                    .with_attention_type(LlamaAttentionType::Unspecified)
                    .with_n_ctx(NonZeroU32::new(1)),
                ModelOutputTraits {
                    causal_attention: true,
                    ..A_NON_CAUSAL_MODEL
                }
            ),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn a_context_without_a_size_takes_the_training_context_size() {
        assert_eq!(
            sequence_output_of(
                &two_sequences()
                    .with_attention_type(LlamaAttentionType::Causal)
                    .with_n_ctx(None),
                ModelOutputTraits {
                    n_ctx_train: 1,
                    ..A_NON_CAUSAL_MODEL
                }
            ),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }
}
