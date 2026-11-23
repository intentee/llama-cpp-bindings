use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::llama_token_data::LlamaTokenData;
use llama_cpp_model::llama_token_data_array::LlamaTokenDataArray;
use llama_cpp_sampler::llama_sampler::LlamaSampler;

#[test]
fn sampler_chain_applies_without_initialized_backend() {
    let sampler_chain = LlamaSampler::chain([LlamaSampler::greedy().unwrap()]).unwrap();
    let mut candidates = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
        ],
        false,
    );

    assert_eq!(sampler_chain.apply(&mut candidates), Ok(()));
    assert_eq!(candidates.selected_token(), Some(LlamaToken::new(1)));
}
