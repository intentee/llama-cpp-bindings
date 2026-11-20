use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::llama_token_data::LlamaTokenData;
use llama_cpp_model::llama_token_data_array::LlamaTokenDataArray;
use llama_cpp_sampler::llama_sampler::LlamaSampler;

/// # Errors
/// Fails when `sampler` cannot be applied to the candidates.
pub fn logits_left_by(sampler: &LlamaSampler, candidates: &[LlamaToken]) -> Result<Vec<f32>> {
    let mut token_data_array = LlamaTokenDataArray::new(
        candidates
            .iter()
            .map(|candidate| LlamaTokenData::new(*candidate, 0.0, 0.0))
            .collect(),
        false,
    );

    sampler
        .apply(&mut token_data_array)
        .context("the sampler must apply to the candidates")
        .map(|()| {
            token_data_array
                .data
                .iter()
                .map(LlamaTokenData::logit)
                .collect()
        })
}
