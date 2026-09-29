use crate::context::LlamaContext;

use super::mtmd_eval_error::MtmdEvalError;

const fn micro_batch_tokens_within(
    requested_batch: i32,
    context_micro_batch: u32,
) -> Result<u32, MtmdEvalError> {
    if requested_batch <= 0 {
        return Err(MtmdEvalError::NonPositiveBatchSize {
            requested: requested_batch,
        });
    }

    let batch_tokens = requested_batch.cast_unsigned();

    Ok(if batch_tokens < context_micro_batch {
        batch_tokens
    } else {
        context_micro_batch
    })
}

/// Returns how many tokens a single decode of `n_batch` evaluates in `llama_ctx`: the smaller of
/// `n_batch` and the context's micro batch.
///
/// # Errors
///
/// Returns [`MtmdEvalError::NonPositiveBatchSize`] when `n_batch` is not positive.
pub fn micro_batch_tokens(llama_ctx: &LlamaContext, n_batch: i32) -> Result<u32, MtmdEvalError> {
    micro_batch_tokens_within(n_batch, llama_ctx.n_ubatch())
}

#[cfg(test)]
mod tests {
    use super::micro_batch_tokens_within;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;

    #[test]
    fn a_batch_smaller_than_the_micro_batch_bounds_a_single_decode() {
        assert_eq!(micro_batch_tokens_within(64, 512), Ok(64));
    }

    #[test]
    fn the_micro_batch_bounds_a_single_decode_of_a_larger_batch() {
        assert_eq!(micro_batch_tokens_within(2048, 512), Ok(512));
    }

    #[test]
    fn a_non_positive_batch_is_rejected() {
        assert_eq!(
            micro_batch_tokens_within(0, 512),
            Err(MtmdEvalError::NonPositiveBatchSize { requested: 0 })
        );
    }
}
