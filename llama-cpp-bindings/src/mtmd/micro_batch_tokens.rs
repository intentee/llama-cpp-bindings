use std::num::NonZeroU32;

use crate::context::LlamaContext;

/// Returns how many tokens a single decode of `n_batch` evaluates in `llama_ctx`: the smaller of
/// `n_batch` and the context's micro batch.
#[must_use]
pub fn micro_batch_tokens(llama_ctx: &LlamaContext, n_batch: NonZeroU32) -> u32 {
    n_batch.get().min(llama_ctx.n_ubatch())
}
