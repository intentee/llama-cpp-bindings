use crate::LlamaContextLoadError;
use crate::context::params::LlamaAttentionType;

pub struct SequenceOutputCapacity {
    pub attention_type: LlamaAttentionType,
    pub model_causal_attention: bool,
    pub model_has_encoder: bool,
    pub model_n_ctx_train: u32,
    pub n_batch: u32,
    pub n_ctx: u32,
    pub n_outputs_max: u32,
    pub n_seq_max: u32,
}

impl SequenceOutputCapacity {
    /// # Errors
    /// Returns [`LlamaContextLoadError::SequencesExceedOutputCapacity`] when there are more
    /// sequences than one batch can output.
    pub fn validate(&self) -> Result<(), LlamaContextLoadError> {
        let n_ctx = if self.n_ctx == 0 {
            self.model_n_ctx_train
        } else {
            self.n_ctx
        };
        let causal_attention = match self.attention_type {
            LlamaAttentionType::Unspecified => self.model_causal_attention,
            LlamaAttentionType::Causal => true,
            LlamaAttentionType::NonCausal => false,
        };
        let n_batch = if causal_attention {
            n_ctx.min(self.n_batch)
        } else {
            self.n_batch
        };
        let output_capacity = if self.n_outputs_max == 0 || self.model_has_encoder {
            n_batch
        } else {
            self.n_outputs_max
        };
        let n_seq_max = self.n_seq_max.max(1);

        if n_seq_max > output_capacity {
            return Err(LlamaContextLoadError::SequencesExceedOutputCapacity {
                n_seq_max,
                output_capacity,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SequenceOutputCapacity;
    use crate::LlamaContextLoadError;
    use crate::context::params::LlamaAttentionType;

    const TWO_SEQUENCES_PAST_ONE_OUTPUT: Result<(), LlamaContextLoadError> =
        Err(LlamaContextLoadError::SequencesExceedOutputCapacity {
            n_seq_max: 2,
            output_capacity: 1,
        });

    fn two_sequences() -> SequenceOutputCapacity {
        SequenceOutputCapacity {
            attention_type: LlamaAttentionType::NonCausal,
            model_causal_attention: false,
            model_has_encoder: false,
            model_n_ctx_train: 4096,
            n_batch: 2,
            n_ctx: 4096,
            n_outputs_max: 0,
            n_seq_max: 2,
        }
    }

    #[test]
    fn the_default_output_capacity_is_one_batch() {
        assert_eq!(
            SequenceOutputCapacity {
                n_batch: 1,
                ..two_sequences()
            }
            .validate(),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn an_explicit_output_capacity_admits_more_sequences_than_one_batch() {
        assert_eq!(
            SequenceOutputCapacity {
                n_batch: 1,
                n_outputs_max: 2,
                ..two_sequences()
            }
            .validate(),
            Ok(())
        );
    }

    #[test]
    fn an_encoder_model_outputs_at_most_one_batch() {
        assert_eq!(
            SequenceOutputCapacity {
                model_has_encoder: true,
                n_batch: 1,
                n_outputs_max: 2,
                ..two_sequences()
            }
            .validate(),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn a_causal_context_outputs_at_most_its_context_size() {
        assert_eq!(
            SequenceOutputCapacity {
                attention_type: LlamaAttentionType::Causal,
                n_ctx: 1,
                ..two_sequences()
            }
            .validate(),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn a_non_causal_context_outputs_a_whole_batch_whatever_its_context_size() {
        assert_eq!(
            SequenceOutputCapacity {
                model_causal_attention: true,
                n_ctx: 1,
                ..two_sequences()
            }
            .validate(),
            Ok(())
        );
    }

    #[test]
    fn an_unspecified_attention_type_follows_the_model() {
        assert_eq!(
            SequenceOutputCapacity {
                attention_type: LlamaAttentionType::Unspecified,
                model_causal_attention: true,
                n_ctx: 1,
                ..two_sequences()
            }
            .validate(),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }

    #[test]
    fn a_context_without_a_size_takes_the_training_context_size() {
        assert_eq!(
            SequenceOutputCapacity {
                attention_type: LlamaAttentionType::Causal,
                model_n_ctx_train: 1,
                n_ctx: 0,
                ..two_sequences()
            }
            .validate(),
            TWO_SEQUENCES_PAST_ONE_OUTPUT
        );
    }
}
