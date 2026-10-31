use llama_cpp_bindings_sys::{llama_pos, llama_rs_batch_entry, llama_seq_id};

use crate::batch_add_error::BatchAddError;
use crate::context::context_limits::ContextLimits;
use crate::sampled_token::SampledToken;
use crate::token::LlamaToken;

/// Tokens to decode together. Each entry names its sequences by a range of `seq_ids`, the layout
/// llama.cpp receives the whole batch in, in one call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaBatch {
    limits: ContextLimits,
    entries: Vec<llama_rs_batch_entry>,
    seq_ids: Vec<llama_seq_id>,
    output_token_indices: Vec<i32>,
    n_tokens: i32,
}

impl LlamaBatch {
    #[must_use]
    pub const fn new(limits: ContextLimits) -> Self {
        Self {
            limits,
            entries: Vec::new(),
            seq_ids: Vec::new(),
            output_token_indices: Vec::new(),
            n_tokens: 0,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.seq_ids.clear();
        self.output_token_indices.clear();
        self.n_tokens = 0;
    }

    /// # Errors
    ///
    /// Returns [`BatchAddError`] when the batch already holds as many tokens as one decode of the
    /// context takes, the token is outside the vocabulary, `seq_ids` is empty or names a sequence
    /// the context does not have, or the token count would overflow.
    pub fn add(
        &mut self,
        sampled_token: &SampledToken,
        pos: llama_pos,
        seq_ids: &[llama_seq_id],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.n_tokens.unsigned_abs() >= self.limits.n_batch {
            return Err(BatchAddError::BatchFull {
                n_batch: self.limits.n_batch,
            });
        }

        let token = self
            .limits
            .ensure_token_in_vocabulary(sampled_token.token())?;

        if seq_ids.is_empty() {
            return Err(BatchAddError::NoSequenceIds);
        }

        for seq_id in seq_ids {
            self.limits.ensure_sequence_id(*seq_id)?;
        }

        let token_index = self.n_tokens;
        self.n_tokens = token_index
            .checked_add(1)
            .ok_or(BatchAddError::TokenCountOverflow {
                n_tokens: token_index,
            })?;

        if logits {
            self.output_token_indices.push(token_index);
        }

        self.entries.push(llama_rs_batch_entry {
            token: token.0,
            pos,
            seq_ids_offset: self.seq_ids.len(),
            seq_ids_count: seq_ids.len(),
            output: logits,
        });
        self.seq_ids.extend_from_slice(seq_ids);

        Ok(())
    }

    /// Adds `tokens` at positions counting from zero in `seq_id`, with logits for the last one,
    /// or for every one when `logits_all` is set.
    ///
    /// # Errors
    ///
    /// Returns the [`BatchAddError`] of the first token [`Self::add`] refuses.
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: llama_seq_id,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        for ((token_index, token), pos) in tokens.iter().enumerate().zip(0..) {
            self.add(
                &SampledToken::Content(*token),
                pos,
                &[seq_id],
                logits_all || token_index + 1 == tokens.len(),
            )?;
        }

        Ok(())
    }

    #[must_use]
    pub const fn limits(&self) -> ContextLimits {
        self.limits
    }

    #[must_use]
    pub const fn n_tokens(&self) -> i32 {
        self.n_tokens
    }

    #[must_use]
    pub fn entries(&self) -> &[llama_rs_batch_entry] {
        &self.entries
    }

    #[must_use]
    pub fn seq_ids(&self) -> &[llama_seq_id] {
        &self.seq_ids
    }

    #[must_use]
    pub fn output_token_indices(&self) -> &[i32] {
        &self.output_token_indices
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_sys::llama_rs_batch_entry;

    use crate::batch_add_error::BatchAddError;
    use crate::context::context_limits::ContextLimits;
    use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;
    use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;

    use super::LlamaBatch;

    const LIMITS: ContextLimits = ContextLimits {
        n_batch: 4,
        n_ubatch: 4,
        n_seq_max: 2,
        n_vocab: 100,
    };

    fn content(token_id: i32) -> SampledToken {
        SampledToken::Content(LlamaToken::new(token_id))
    }

    #[test]
    fn a_batch_whose_token_count_reached_its_type_limit_refuses_another_token() {
        let mut batch = LlamaBatch {
            n_tokens: i32::MAX,
            ..LlamaBatch::new(ContextLimits {
                n_batch: u32::MAX,
                ..LIMITS
            })
        };

        assert_eq!(
            batch.add(&content(1), 0, &[0], false),
            Err(BatchAddError::TokenCountOverflow { n_tokens: i32::MAX })
        );
    }

    #[test]
    fn added_tokens_are_kept_in_order_with_their_outputs() {
        let mut batch = LlamaBatch::new(LIMITS);

        batch.add(&content(1), 0, &[0], false).unwrap();
        batch.add(&content(2), 1, &[0, 1], true).unwrap();

        assert_eq!(
            batch.entries(),
            [
                llama_rs_batch_entry {
                    token: 1,
                    pos: 0,
                    seq_ids_offset: 0,
                    seq_ids_count: 1,
                    output: false,
                },
                llama_rs_batch_entry {
                    token: 2,
                    pos: 1,
                    seq_ids_offset: 1,
                    seq_ids_count: 2,
                    output: true,
                },
            ]
        );
        assert_eq!(batch.seq_ids(), [0, 0, 1]);
        assert_eq!(batch.output_token_indices(), [1]);
        assert_eq!(batch.n_tokens(), 2);
    }

    #[test]
    fn clearing_forgets_every_token_and_output() {
        let mut batch = LlamaBatch::new(LIMITS);
        batch.add(&content(1), 0, &[0], true).unwrap();

        batch.clear();

        assert_eq!(batch, LlamaBatch::new(LIMITS));
    }

    #[test]
    fn a_batch_holding_one_decode_of_tokens_refuses_another() {
        let mut batch = LlamaBatch::new(LIMITS);
        batch
            .add_sequence(&[1, 2, 3, 4].map(LlamaToken::new), 0, false)
            .unwrap();

        assert_eq!(
            batch.add(&content(5), 4, &[0], false),
            Err(BatchAddError::BatchFull { n_batch: 4 })
        );
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(&content(100), 0, &[0], false),
            Err(BatchAddError::TokenOutsideVocabulary(
                TokenOutsideVocabulary {
                    token: LlamaToken::new(100),
                    n_vocab: 100,
                }
            ))
        );
    }

    #[test]
    fn a_token_without_sequences_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(&content(1), 0, &[], false),
            Err(BatchAddError::NoSequenceIds)
        );
    }

    #[test]
    fn a_token_in_a_sequence_the_context_lacks_is_refused_and_not_added() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(&content(1), 0, &[0, 2], false),
            Err(BatchAddError::SequenceIdOutOfRange(SequenceIdOutOfRange {
                seq_id: 2,
                n_seq_max: 2,
            }))
        );
        assert_eq!(batch, LlamaBatch::new(LIMITS));
    }

    #[test]
    fn a_sequence_requests_logits_for_its_last_token_only() {
        let mut batch = LlamaBatch::new(LIMITS);

        batch
            .add_sequence(&[10, 20, 30].map(LlamaToken::new), 1, false)
            .unwrap();

        assert_eq!(batch.output_token_indices(), [2]);
        assert_eq!(
            batch.entries().last(),
            Some(&llama_rs_batch_entry {
                token: 30,
                pos: 2,
                seq_ids_offset: 2,
                seq_ids_count: 1,
                output: true,
            })
        );
    }

    #[test]
    fn a_sequence_with_logits_for_all_requests_every_token() {
        let mut batch = LlamaBatch::new(LIMITS);

        batch
            .add_sequence(&[10, 20, 30].map(LlamaToken::new), 0, true)
            .unwrap();

        assert_eq!(batch.output_token_indices(), [0, 1, 2]);
    }

    #[test]
    fn a_sequence_longer_than_one_decode_stops_at_the_first_refused_token() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add_sequence(&[1, 2, 3, 4, 5].map(LlamaToken::new), 0, false),
            Err(BatchAddError::BatchFull { n_batch: 4 })
        );
        assert_eq!(batch.n_tokens(), 4);
    }
}
