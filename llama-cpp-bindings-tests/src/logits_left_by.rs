use anyhow::Result;

use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings::token::data::LlamaTokenData;
use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;

/// # Errors
/// Forwards [`LlamaSampler::apply`] errors verbatim.
pub fn logits_left_by(sampler: &LlamaSampler, candidates: &[LlamaToken]) -> Result<Vec<f32>> {
    let mut token_data_array = LlamaTokenDataArray::new(
        candidates
            .iter()
            .map(|candidate| LlamaTokenData::new(*candidate, 0.0, 0.0))
            .collect(),
        false,
    );

    sampler.apply(&mut token_data_array)?;

    Ok(token_data_array
        .data
        .iter()
        .map(LlamaTokenData::logit)
        .collect())
}
