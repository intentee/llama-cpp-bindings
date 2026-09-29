use std::num::NonZeroU32;

use super::mtmd_eval_error::MtmdEvalError;

/// Returns `n_batch` as a positive token count.
///
/// # Errors
///
/// Returns [`MtmdEvalError::NonPositiveBatchSize`] when `n_batch` is not positive.
pub fn positive_batch_tokens(n_batch: i32) -> Result<NonZeroU32, MtmdEvalError> {
    u32::try_from(n_batch)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or(MtmdEvalError::NonPositiveBatchSize { requested: n_batch })
}

#[cfg(test)]
mod tests {
    use super::positive_batch_tokens;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;

    #[test]
    fn a_zero_batch_is_rejected() {
        assert_eq!(
            positive_batch_tokens(0),
            Err(MtmdEvalError::NonPositiveBatchSize { requested: 0 })
        );
    }

    #[test]
    fn a_negative_batch_is_rejected() {
        assert_eq!(
            positive_batch_tokens(-1),
            Err(MtmdEvalError::NonPositiveBatchSize { requested: -1 })
        );
    }
}
