use std::num::NonZeroU32;

use llama_cpp_context::context_limits::ContextLimits;
use llama_cpp_context::llama_context::LlamaContext;

use crate::mtmd_error::MtmdError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtmdEvalLimits {
    pub context: ContextLimits,
    pub decodes_batches_in_one_micro_batch: bool,
}

impl MtmdEvalLimits {
    #[must_use]
    pub fn of(llama_ctx: &LlamaContext) -> Self {
        Self {
            context: llama_ctx.limits(),
            decodes_batches_in_one_micro_batch: llama_ctx.decodes_batches_in_one_micro_batch(),
        }
    }

    /// Returns `n_batch` as a token count once both it and `seq_id` fit the context, so the mtmd
    /// helper never adds a token its batch refuses.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::NonPositiveBatchSize`] when `n_batch` is not positive,
    /// [`MtmdError::BatchSizeExceedsContextLimit`] when it exceeds the context's batch size,
    /// or [`MtmdError::SequenceIdOutOfRange`] when `seq_id` names no sequence of the context.
    pub fn batch_tokens(
        &self,
        n_batch: i32,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
    ) -> Result<NonZeroU32, MtmdError> {
        let batch_tokens = u32::try_from(n_batch)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(MtmdError::NonPositiveBatchSize { requested: n_batch })?;

        if batch_tokens.get() > self.context.batch {
            return Err(MtmdError::BatchSizeExceedsContextLimit {
                requested: n_batch,
                context_max: self.context.batch,
            });
        }

        if !self.context.holds_sequence(seq_id) {
            return Err(MtmdError::SequenceIdOutOfRange {
                seq_id,
                n_seq_max: self.context.max_sequences,
            });
        }

        Ok(batch_tokens)
    }

    /// Returns how many tokens each decode of `n_batch` evaluates in the context, once
    /// [`Self::batch_tokens`] accepts `n_batch` and `seq_id`. The mtmd helper decodes text and
    /// causally decoded media in pieces of up to `n_batch` tokens, which a context that decodes
    /// every batch in a single micro batch aborts on once they exceed that micro batch.
    ///
    /// # Errors
    ///
    /// Returns any error of [`Self::batch_tokens`], or
    /// [`MtmdError::BatchSizeExceedsSingleMicroBatch`] when the context decodes every batch in a
    /// single micro batch and `n_batch` exceeds it.
    pub fn micro_batch_tokens(
        &self,
        n_batch: i32,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
    ) -> Result<u32, MtmdError> {
        let batch_tokens = self.batch_tokens(n_batch, seq_id)?;

        if self.decodes_batches_in_one_micro_batch && batch_tokens.get() > self.context.micro_batch
        {
            return Err(MtmdError::BatchSizeExceedsSingleMicroBatch {
                requested: n_batch,
                micro_batch: self.context.micro_batch,
            });
        }

        Ok(self.context.micro_batch_tokens(batch_tokens))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use llama_cpp_context::context_limits::ContextLimits;

    use crate::mtmd_error::MtmdError;
    use crate::mtmd_eval_limits::MtmdEvalLimits;

    const LIMITS: MtmdEvalLimits = MtmdEvalLimits {
        context: ContextLimits {
            batch: 512,
            micro_batch: 128,
            max_sequences: 2,
            vocabulary_size: 100,
        },
        decodes_batches_in_one_micro_batch: false,
    };

    const SINGLE_MICRO_BATCH_LIMITS: MtmdEvalLimits = MtmdEvalLimits {
        decodes_batches_in_one_micro_batch: true,
        ..LIMITS
    };

    #[test]
    fn a_batch_within_the_context_for_an_existing_sequence_is_accepted() {
        assert_eq!(
            LIMITS.batch_tokens(512, 1),
            Ok(NonZeroU32::new(512).unwrap())
        );
    }

    #[test]
    fn a_zero_batch_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(0, 0),
            Err(MtmdError::NonPositiveBatchSize { requested: 0 })
        );
    }

    #[test]
    fn a_negative_batch_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(-1, 0),
            Err(MtmdError::NonPositiveBatchSize { requested: -1 })
        );
    }

    #[test]
    fn a_batch_larger_than_the_context_batch_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(513, 0),
            Err(MtmdError::BatchSizeExceedsContextLimit {
                requested: 513,
                context_max: 512,
            })
        );
    }

    #[test]
    fn an_accepted_batch_decodes_in_the_context_micro_batch_it_exceeds() {
        assert_eq!(LIMITS.micro_batch_tokens(512, 1), Ok(128));
    }

    #[test]
    fn a_refused_batch_has_no_micro_batch() {
        assert_eq!(
            LIMITS.micro_batch_tokens(0, 0),
            Err(MtmdError::NonPositiveBatchSize { requested: 0 })
        );
    }

    #[test]
    fn a_batch_larger_than_the_single_micro_batch_of_the_context_is_rejected() {
        assert_eq!(
            SINGLE_MICRO_BATCH_LIMITS.micro_batch_tokens(129, 0),
            Err(MtmdError::BatchSizeExceedsSingleMicroBatch {
                requested: 129,
                micro_batch: 128,
            })
        );
    }

    #[test]
    fn a_batch_within_the_single_micro_batch_of_the_context_decodes_whole() {
        assert_eq!(
            SINGLE_MICRO_BATCH_LIMITS.micro_batch_tokens(128, 0),
            Ok(128)
        );
    }

    #[test]
    fn a_sequence_id_the_context_lacks_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(64, 2),
            Err(MtmdError::SequenceIdOutOfRange {
                seq_id: 2,
                n_seq_max: 2,
            })
        );
    }
}
