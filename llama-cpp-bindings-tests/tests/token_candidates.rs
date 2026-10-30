use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings::token::data::LlamaTokenData;
use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn token_data_array_has_entries_after_decode(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let token_data_array = context.token_data_array()?;

    assert_ne!(token_data_array.data, Vec::<LlamaTokenData>::new());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn token_data_array_ith_returns_valid_data(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let token_data_array = context.token_data_array_ith(last_index)?;

    assert_eq!(
        token_data_array.data.len(),
        usize::try_from(fixture.model.n_vocab())?
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn candidates_returns_n_vocab_entries(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let count = context.candidates()?.count();

    assert_eq!(count, usize::try_from(fixture.model.n_vocab())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn candidates_ith_returns_n_vocab_entries(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let count = context.candidates_ith(last_index)?.count();

    assert_eq!(count, usize::try_from(fixture.model.n_vocab())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_refilled_candidate_array_matches_a_freshly_built_one(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let prompt_tokens = model.str_to_token("Hello", AddBos::Never, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&prompt_tokens, 0, false)?;
    context.decode(&batch)?;

    let last_index = batch.n_tokens() - 1;
    let mut reused_candidates = LlamaTokenDataArray::new(
        vec![LlamaTokenData::new(LlamaToken::new(7), 1.0, 0.5)],
        true,
    );
    reused_candidates.selected = Some(0);

    context.fill_token_data_array_ith(last_index, &mut reused_candidates)?;

    assert_eq!(reused_candidates, context.token_data_array_ith(last_index)?);

    Ok(())
}
