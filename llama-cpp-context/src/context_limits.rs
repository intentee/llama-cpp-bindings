use std::num::NonZeroU32;

use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_bindings_sys::bindings::llama_seq_id;

const ATTENTION_MASK_REUSE_MARGIN: u32 = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextLimits {
    pub batch: u32,
    pub micro_batch: u32,
    pub max_sequences: u32,
    pub vocabulary_size: i32,
    pub sliding_window: u32,
}

impl ContextLimits {
    /// Whether `seq_id` names one of the sequences of the context.
    #[must_use]
    pub fn holds_sequence(&self, seq_id: llama_seq_id) -> bool {
        u32::try_from(seq_id).is_ok_and(|sequence| sequence < self.max_sequences)
    }

    /// The largest position a token of the context can take. While it decodes, llama.cpp adds to
    /// the positions a sequence holds in `llama_pos` arithmetic that must not overflow: building
    /// the attention mask adds the sliding window plus 32 to every cached position, and a
    /// recurrent memory adds up to a micro batch of tokens to the last position of the sequence.
    #[must_use]
    pub fn max_position(&self) -> llama_pos {
        let attention_mask_reach = self
            .sliding_window
            .saturating_add(ATTENTION_MASK_REUSE_MARGIN);

        llama_pos::MAX.saturating_sub_unsigned(attention_mask_reach.max(self.micro_batch))
    }

    /// Returns how many tokens a single decode of `batch_tokens` evaluates: the smaller of
    /// `batch_tokens` and the context's micro batch.
    #[must_use]
    pub const fn micro_batch_tokens(&self, batch_tokens: NonZeroU32) -> u32 {
        let batch_tokens = batch_tokens.get();

        if batch_tokens < self.micro_batch {
            batch_tokens
        } else {
            self.micro_batch
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::ContextLimits;

    const LIMITS: ContextLimits = ContextLimits {
        batch: 512,
        micro_batch: 128,
        max_sequences: 2,
        vocabulary_size: 100,
        sliding_window: 0,
    };

    #[test]
    fn the_last_sequence_of_the_context_is_held() {
        assert!(LIMITS.holds_sequence(1));
    }

    #[test]
    fn a_negative_sequence_id_is_not_held() {
        assert!(!LIMITS.holds_sequence(-1));
    }

    #[test]
    fn a_sequence_id_past_the_last_sequence_is_not_held() {
        assert!(!LIMITS.holds_sequence(2));
    }

    #[test]
    fn a_micro_batch_wider_than_the_attention_mask_reach_bounds_the_max_position() {
        assert_eq!(LIMITS.max_position(), i32::MAX - 128);
    }

    #[test]
    fn a_sliding_window_whose_mask_reach_exceeds_the_micro_batch_bounds_the_max_position() {
        assert_eq!(
            ContextLimits {
                sliding_window: 4096,
                ..LIMITS
            }
            .max_position(),
            i32::MAX - 4128
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
