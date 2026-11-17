use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::llama_token_data::LlamaTokenData;
use llama_cpp_model::llama_token_data_array::LlamaTokenDataArray;

#[test]
fn sampler_chain_applies_without_initialized_backend() -> Result<()> {
    let sampler_chain = LlamaSampler::chain([LlamaSampler::greedy()?])?;
    let mut candidates = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
        ],
        false,
    );

    sampler_chain.apply(&mut candidates)?;

    ensure!(candidates.selected_token() == Some(LlamaToken::new(1)));

    Ok(())
}
