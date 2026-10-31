use std::ffi::c_int;
use std::num::NonZeroI32;

use crate::error::batch_transfer_error::BatchTransferError;
use crate::error::process_failure::ProcessFailure;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    #[error(transparent)]
    Process(#[from] ProcessFailure),
    #[error("model has no encoder")]
    ModelHasNoEncoder,
    #[error("no KV cache slot was available")]
    NoKvCacheSlot,
    #[error("encode batch is invalid (empty or initialization failure)")]
    BatchInvalid,
    #[error("encode ran out of memory")]
    EncodeOutOfMemory,
    #[error("backend compute failed during encode")]
    ComputeFailed,
    #[error("encode returned an unknown status code: {code}")]
    UnknownStatus { code: c_int },
    #[error("encode batch of {n_tokens} tokens must fit a single micro batch of {n_ubatch} tokens")]
    BatchExceedsMicroBatch { n_tokens: i32, n_ubatch: u32 },
    #[error("the batch could not be handed to llama.cpp: {0}")]
    BatchTransfer(#[from] BatchTransferError),
}

impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => Self::NoKvCacheSlot,
            -1 => Self::BatchInvalid,
            -2 => Self::EncodeOutOfMemory,
            -3 => Self::ComputeFailed,
            error_code => Self::UnknownStatus { code: error_code },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use crate::error::encode_error::EncodeError;

    #[test]
    fn no_kv_cache_slot_maps_from_code_one() {
        let error = EncodeError::from(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, EncodeError::NoKvCacheSlot);
    }

    #[test]
    fn batch_invalid_maps_from_code_negative_one() {
        let error = EncodeError::from(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, EncodeError::BatchInvalid);
    }

    #[test]
    fn allocation_and_compute_failures_map_from_codes_negative_two_and_three() {
        assert_eq!(
            [-2, -3].map(|code| EncodeError::from(NonZeroI32::new(code).expect("non-zero code"))),
            [EncodeError::EncodeOutOfMemory, EncodeError::ComputeFailed]
        );
    }

    #[test]
    fn unrecognized_code_falls_through_to_unknown_status() {
        let error = EncodeError::from(NonZeroI32::new(99).expect("99 is non-zero"));

        assert_eq!(error, EncodeError::UnknownStatus { code: 99 });
    }
}
