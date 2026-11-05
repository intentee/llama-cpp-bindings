use std::num::NonZeroU32;

use super::mtmd_eval_error::MtmdEvalError;
use crate::context::LlamaContext;
use crate::context::context_limits::ContextLimits;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtmdEvalLimits {
    pub context: ContextLimits,
}

impl MtmdEvalLimits {
    #[must_use]
    pub const fn of(llama_ctx: &LlamaContext) -> Self {
        Self {
            context: ContextLimits::of(llama_ctx),
        }
    }

    /// Returns `n_batch` as a token count once both it and `seq_id` fit the context, so the mtmd
    /// helper never adds a token its batch refuses.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdEvalError::NonPositiveBatchSize`] when `n_batch` is not positive,
    /// [`MtmdEvalError::BatchSizeExceedsContextLimit`] when it exceeds the context's batch size,
    /// or [`MtmdEvalError::SequenceIdOutOfRange`] when `seq_id` names no sequence of the context.
    pub fn batch_tokens(
        &self,
        n_batch: i32,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
    ) -> Result<NonZeroU32, MtmdEvalError> {
        let batch_tokens = u32::try_from(n_batch)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(MtmdEvalError::NonPositiveBatchSize { requested: n_batch })?;

        if batch_tokens.get() > self.context.n_batch {
            return Err(MtmdEvalError::BatchSizeExceedsContextLimit {
                requested: n_batch,
                context_max: self.context.n_batch,
            });
        }

        self.context.ensure_sequence_id(seq_id)?;

        Ok(batch_tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::context::context_limits::ContextLimits;
    use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;
    use crate::mtmd::mtmd_eval_limits::MtmdEvalLimits;

    const LIMITS: MtmdEvalLimits = MtmdEvalLimits {
        context: ContextLimits {
            n_batch: 512,
            n_ubatch: 128,
            n_seq_max: 2,
            n_vocab: 100,
        },
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
            Err(MtmdEvalError::NonPositiveBatchSize { requested: 0 })
        );
    }

    #[test]
    fn a_negative_batch_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(-1, 0),
            Err(MtmdEvalError::NonPositiveBatchSize { requested: -1 })
        );
    }

    #[test]
    fn a_batch_larger_than_the_context_batch_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(513, 0),
            Err(MtmdEvalError::BatchSizeExceedsContextLimit {
                requested: 513,
                context_max: 512,
            })
        );
    }

    #[test]
    fn a_sequence_id_the_context_lacks_is_rejected() {
        assert_eq!(
            LIMITS.batch_tokens(64, 2),
            Err(MtmdEvalError::SequenceIdOutOfRange(SequenceIdOutOfRange {
                seq_id: 2,
                n_seq_max: 2,
            }))
        );
    }
}
