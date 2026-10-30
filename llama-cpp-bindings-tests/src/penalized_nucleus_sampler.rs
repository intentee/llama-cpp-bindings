use anyhow::Result;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::sampling::LlamaSampler;

const SAMPLING_SEED: u32 = 0x00C0_FFEE;

/// A seeded sampler that penalizes repetition and samples from the nucleus, so a small model
/// does not loop while its answer stays reproducible.
///
/// # Errors
///
/// Forwards sampler construction errors.
pub fn penalized_nucleus_sampler(model: &LlamaModel) -> Result<LlamaSampler> {
    Ok(LlamaSampler::chain_simple([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0)?,
        LlamaSampler::top_k(40)?,
        LlamaSampler::top_p(0.9, 1)?,
        LlamaSampler::min_p(0.05, 1)?,
        LlamaSampler::temp(0.7)?,
        LlamaSampler::dist(SAMPLING_SEED)?,
    ])?)
}
