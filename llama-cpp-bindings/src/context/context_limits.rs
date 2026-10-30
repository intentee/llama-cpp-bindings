use std::num::NonZeroU32;

use llama_cpp_bindings_sys::llama_seq_id;

use crate::context::LlamaContext;
use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;
use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;
use crate::token::LlamaToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextLimits {
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_vocab: i32,
}

impl ContextLimits {
    #[must_use]
    pub fn of(context: &LlamaContext) -> Self {
        Self {
            n_batch: context.n_batch(),
            n_ubatch: context.n_ubatch(),
            n_seq_max: context.n_seq_max(),
            n_vocab: context.model.n_vocab(),
        }
    }

    /// # Errors
    ///
    /// Returns [`TokenOutsideVocabulary`] when `token` is not in the vocabulary of the model.
    pub fn ensure_token_in_vocabulary(
        &self,
        token: LlamaToken,
    ) -> Result<LlamaToken, TokenOutsideVocabulary> {
        TokenOutsideVocabulary::check(token, self.n_vocab)
    }

    /// # Errors
    ///
    /// Returns [`SequenceIdOutOfRange`] when `seq_id` names no sequence of the context.
    pub fn ensure_sequence_id(&self, seq_id: llama_seq_id) -> Result<(), SequenceIdOutOfRange> {
        if u32::try_from(seq_id).is_ok_and(|sequence| sequence < self.n_seq_max) {
            Ok(())
        } else {
            Err(SequenceIdOutOfRange {
                seq_id,
                n_seq_max: self.n_seq_max,
            })
        }
    }

    /// Returns how many tokens a single decode of `batch_tokens` evaluates: the smaller of
    /// `batch_tokens` and the context's micro batch.
    #[must_use]
    pub const fn micro_batch_tokens(&self, batch_tokens: NonZeroU32) -> u32 {
        let batch_tokens = batch_tokens.get();

        if batch_tokens < self.n_ubatch {
            batch_tokens
        } else {
            self.n_ubatch
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::ContextLimits;
    use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;
    use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;
    use crate::token::LlamaToken;

    const LIMITS: ContextLimits = ContextLimits {
        n_batch: 512,
        n_ubatch: 128,
        n_seq_max: 2,
        n_vocab: 100,
    };

    #[test]
    fn the_last_sequence_of_the_context_is_accepted() {
        assert_eq!(LIMITS.ensure_sequence_id(1), Ok(()));
    }

    #[test]
    fn a_negative_sequence_id_is_rejected() {
        assert_eq!(
            LIMITS.ensure_sequence_id(-1),
            Err(SequenceIdOutOfRange {
                seq_id: -1,
                n_seq_max: 2,
            })
        );
    }

    #[test]
    fn a_sequence_id_past_the_last_sequence_is_rejected() {
        assert_eq!(
            LIMITS.ensure_sequence_id(2),
            Err(SequenceIdOutOfRange {
                seq_id: 2,
                n_seq_max: 2,
            })
        );
    }

    #[test]
    fn the_last_token_of_the_vocabulary_is_accepted() {
        assert_eq!(
            LIMITS.ensure_token_in_vocabulary(LlamaToken::new(99)),
            Ok(LlamaToken::new(99))
        );
    }

    #[test]
    fn a_token_past_the_vocabulary_is_rejected() {
        assert_eq!(
            LIMITS.ensure_token_in_vocabulary(LlamaToken::new(100)),
            Err(TokenOutsideVocabulary {
                token: LlamaToken::new(100),
                n_vocab: 100,
            })
        );
    }

    #[test]
    fn a_negative_token_is_rejected() {
        assert_eq!(
            LIMITS.ensure_token_in_vocabulary(LlamaToken::new(-1)),
            Err(TokenOutsideVocabulary {
                token: LlamaToken::new(-1),
                n_vocab: 100,
            })
        );
    }

    #[test]
    fn a_batch_smaller_than_the_micro_batch_is_decoded_whole() {
        assert_eq!(LIMITS.micro_batch_tokens(NonZeroU32::new(64).unwrap()), 64);
    }

    #[test]
    fn a_batch_larger_than_the_micro_batch_is_decoded_in_micro_batches() {
        assert_eq!(
            LIMITS.micro_batch_tokens(NonZeroU32::new(256).unwrap()),
            128
        );
    }
}
