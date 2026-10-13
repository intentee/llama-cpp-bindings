use std::ffi::CStr;
use std::ptr::NonNull;
use std::slice;

use crate::token::LlamaToken;

use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk_error::MtmdInputChunkError;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;
use super::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

/// # Safety
///
/// `tokens_ptr` must point to at least `n_tokens` valid `llama_token` values
/// that remain valid for the lifetime `'chunk`.
const unsafe fn tokens_from_raw_ptr<'chunk>(
    tokens_ptr: *const llama_cpp_bindings_sys::llama_token,
    n_tokens: usize,
) -> Option<&'chunk [LlamaToken]> {
    if tokens_ptr.is_null() || n_tokens == 0 {
        None
    } else {
        unsafe {
            Some(slice::from_raw_parts(
                tokens_ptr.cast::<LlamaToken>(),
                n_tokens,
            ))
        }
    }
}

fn non_causal_chunk_micro_batch_error(
    decodes_non_causally: bool,
    chunk_tokens: usize,
    micro_batch_tokens: u32,
) -> Option<MtmdEvalError> {
    if decodes_non_causally
        && u64::try_from(chunk_tokens).is_ok_and(|tokens| tokens > u64::from(micro_batch_tokens))
    {
        return Some(MtmdEvalError::NonCausalChunkExceedsMicroBatch(
            NonCausalChunkMicroBatchMismatch {
                chunk_tokens,
                micro_batch_tokens,
            },
        ));
    }

    None
}

#[derive(Debug)]
pub struct MtmdInputChunk {
    pub chunk: NonNull<llama_cpp_bindings_sys::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdInputChunkTypeError> {
        let chunk_type =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::try_from(chunk_type)
    }

    /// # Errors
    ///
    /// Returns [`MtmdInputChunkTypeError`] when the wrapper reports a chunk type this
    /// binding does not know, so an unclassifiable chunk is never mistaken for a
    /// non-text chunk.
    pub fn text_tokens(&self) -> Result<Option<&[LlamaToken]>, MtmdInputChunkTypeError> {
        Ok(self.text_tokens_of(self.chunk_type()?))
    }

    #[must_use]
    pub fn text_tokens_of(&self, chunk_type: MtmdInputChunkType) -> Option<&[LlamaToken]> {
        if chunk_type != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_bindings_sys::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        unsafe { tokens_from_raw_ptr(tokens_ptr, n_tokens) }
    }

    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::ChunkOperationFailed` if copying fails.
    pub fn copy(&self) -> Result<Self, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::ChunkOperationFailed)?;

        Ok(Self { chunk, owned: true })
    }

    /// Checks that this chunk can be evaluated in decodes of `micro_batch_tokens` and returns the
    /// chunk type it classified. llama.cpp splits a causal chunk across decodes, while a media
    /// chunk it decodes non-causally has to fit a single decode.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdEvalError::NonCausalChunkExceedsMicroBatch`] when a media chunk decoded
    /// non-causally has more tokens than `micro_batch_tokens`, or
    /// [`MtmdEvalError::UnknownChunkType`] when the chunk type is unknown.
    pub fn fit_to_micro_batch(
        &self,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<MtmdInputChunkType, MtmdEvalError> {
        let chunk_type = self.chunk_type()?;
        let decodes_non_causally =
            chunk_type != MtmdInputChunkType::Text && mtmd_ctx.decode_use_non_causal(self);

        non_causal_chunk_micro_batch_error(
            decodes_non_causally,
            self.n_tokens(),
            micro_batch_tokens,
        )
        .map_or(Ok(chunk_type), Err)
    }
}

impl Drop for MtmdInputChunk {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::non_causal_chunk_micro_batch_error;
    use super::tokens_from_raw_ptr;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;
    use crate::mtmd::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_null() {
        assert!(unsafe { tokens_from_raw_ptr(std::ptr::null(), 5) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_zero_count() {
        let token: llama_cpp_bindings_sys::llama_token = 42;
        assert!(unsafe { tokens_from_raw_ptr(&raw const token, 0) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_some_for_valid() {
        let tokens: [llama_cpp_bindings_sys::llama_token; 2] = [1, 2];
        let result = unsafe { tokens_from_raw_ptr(tokens.as_ptr(), 2) };

        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn a_non_causal_chunk_larger_than_the_micro_batch_reports_the_mismatch() {
        assert_eq!(
            non_causal_chunk_micro_batch_error(true, 9, 4),
            Some(MtmdEvalError::NonCausalChunkExceedsMicroBatch(
                NonCausalChunkMicroBatchMismatch {
                    chunk_tokens: 9,
                    micro_batch_tokens: 4,
                }
            ))
        );
    }

    #[test]
    fn a_causal_chunk_larger_than_the_micro_batch_is_split_instead() {
        assert!(non_causal_chunk_micro_batch_error(false, 9, 4).is_none());
    }

    #[test]
    fn a_non_causal_chunk_within_the_micro_batch_fits() {
        assert!(non_causal_chunk_micro_batch_error(true, 4, 4).is_none());
    }
}
