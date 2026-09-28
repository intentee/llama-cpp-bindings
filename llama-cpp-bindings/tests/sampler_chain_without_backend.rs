use anyhow::Result;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings::token::data::LlamaTokenData;
use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;

#[test]
fn sampler_chain_applies_without_initialized_backend() -> Result<()> {
    let sampler_chain = LlamaSampler::chain_simple([LlamaSampler::greedy()?])?;
    let mut candidates = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
        ],
        false,
    );

    candidates.apply_sampler(&sampler_chain)?;

    assert_eq!(candidates.selected_token(), Some(LlamaToken::new(1)));

    Ok(())
}
