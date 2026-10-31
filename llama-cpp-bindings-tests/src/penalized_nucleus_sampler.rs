use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::sampling::LlamaSampler;

const SAMPLING_SEED: u32 = 0x00C0_FFEE;

/// A seeded sampler that penalizes repetition and samples from the nucleus, so a small model
/// does not loop while its answer stays reproducible.
///
/// # Panics
///
/// Panics when llama.cpp cannot build one of the samplers.
#[must_use]
pub fn penalized_nucleus_sampler(model: &LlamaModel) -> LlamaSampler {
    LlamaSampler::chain([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0)
            .expect("the penalties sampler must build"),
        LlamaSampler::top_k(40).expect("the top-k sampler must build"),
        LlamaSampler::top_p(0.9, 1).expect("the top-p sampler must build"),
        LlamaSampler::min_p(0.05, 1).expect("the min-p sampler must build"),
        LlamaSampler::temp(0.7).expect("the temperature sampler must build"),
        LlamaSampler::dist(SAMPLING_SEED).expect("the distribution sampler must build"),
    ])
    .expect("the sampler chain must build")
}
