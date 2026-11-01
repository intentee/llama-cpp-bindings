use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings::token::data::LlamaTokenData;
use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;

/// # Panics
/// Panics when `sampler` cannot be applied to the candidates.
#[must_use]
pub fn logits_left_by(sampler: &LlamaSampler, candidates: &[LlamaToken]) -> Vec<f32> {
    let mut token_data_array = LlamaTokenDataArray::new(
        candidates
            .iter()
            .map(|candidate| LlamaTokenData::new(*candidate, 0.0, 0.0))
            .collect(),
        false,
    );

    sampler
        .apply(&mut token_data_array)
        .expect("the sampler must apply to the candidates");

    token_data_array
        .data
        .iter()
        .map(LlamaTokenData::logit)
        .collect()
}
