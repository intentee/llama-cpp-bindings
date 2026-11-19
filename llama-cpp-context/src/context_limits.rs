use std::num::NonZeroU32;

use llama_cpp_bindings_sys::bindings::llama_seq_id;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextLimits {
    pub batch: u32,
    pub micro_batch: u32,
    pub max_sequences: u32,
    pub vocabulary_size: i32,
}

impl ContextLimits {
    /// Whether `seq_id` names one of the sequences of the context.
    #[must_use]
    pub fn holds_sequence(&self, seq_id: llama_seq_id) -> bool {
        u32::try_from(seq_id).is_ok_and(|sequence| sequence < self.max_sequences)
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
