use anyhow::Result;
use llama_cpp_bindings::EmbeddingsError;
use llama_cpp_bindings::SampledToken;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::params::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_types::BatchCapacity;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const STATE: &str = "Shoes arrived two weeks late and in the wrong size.";
const FIRST_BRANCH: &str = " Which team should handle this ticket?";
const SECOND_BRANCH: &str = " Is the customer angry about the delay?";
const OTHER_STATE: &str = "The weather in Oslo is cold, dark and windy all week.";

fn tokenize(fixture: &LlamaFixture<'_>, text: &str) -> Result<Vec<LlamaToken>> {
    Ok(fixture
        .model
        .str_to_token(text, AddBos::Never, ParseSpecialTokens::Never)?)
}

fn hidden_state_context<'fixture>(
    fixture: &'fixture LlamaFixture<'_>,
) -> Result<LlamaContext<'fixture>> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::None)
            .with_kv_unified(true),
    )?;

    context.enable_masked_nextn_embeddings()?;

    Ok(context)
}

fn add_tokens(
    batch: &mut LlamaBatch<'_>,
    tokens: &[LlamaToken],
    first_position: i32,
    sequence_id: i32,
    output_on_last_token: bool,
) -> Result<()> {
    let last_index = tokens.len() - 1;

    for (index, token) in tokens.iter().enumerate() {
        batch.add(
            &SampledToken::Content(*token),
            first_position + i32::try_from(index)?,
            &[sequence_id],
            output_on_last_token && index == last_index,
        )?;
    }

    Ok(())
}

fn hidden_state_after(
    fixture: &LlamaFixture<'_>,
    state: &[LlamaToken],
    branch: &[LlamaToken],
) -> Result<Vec<f32>> {
    let mut context = hidden_state_context(fixture)?;
    let mut state_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(&mut state_batch, state, 0, 0, false)?;
    context.decode(&mut state_batch)?;

    let mut branch_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(
        &mut branch_batch,
        branch,
        i32::try_from(state.len())?,
        0,
        true,
    )?;
    context.decode(&mut branch_batch)?;

    Ok(context
        .nextn_embeddings_ith(branch_batch.n_tokens() - 1)?
        .to_vec())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn masked_nextn_embeddings_equal_the_final_hidden_states(fixture: &LlamaFixture<'_>) -> Result<()> {
    let tokens = tokenize(fixture, STATE)?;
    let last_index = i32::try_from(tokens.len() - 1)?;

    let mut embedding_context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::None),
    )?;
    let mut embedding_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    embedding_batch.add_sequence(&tokens, 0, true)?;
    embedding_context.decode(&mut embedding_batch)?;

    let mut nextn_context = hidden_state_context(fixture)?;
    let mut nextn_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    nextn_batch.add_sequence(&tokens, 0, true)?;
    nextn_context.decode(&mut nextn_batch)?;

    assert_eq!(
        nextn_context.nextn_embeddings_ith(last_index)?,
        embedding_context.embeddings_ith(last_index)?,
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn masked_nextn_embeddings_exist_only_for_output_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let tokens = tokenize(fixture, STATE)?;
    let mut context = hidden_state_context(fixture)?;
    let mut batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(&mut batch, &tokens, 0, 0, true)?;
    context.decode(&mut batch)?;

    assert_eq!(
        context.nextn_embeddings_ith(0),
        Err(EmbeddingsError::NextnEmbeddingUnavailable { token_index: 0 })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn nextn_embeddings_refuse_a_pooling_context(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::Mean),
    )?;

    assert_eq!(
        context.enable_masked_nextn_embeddings(),
        Err(EmbeddingsError::NextnEmbeddingsRequireNonePooling {
            pooling_type: LlamaPoolingType::Mean,
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn nextn_embeddings_are_unavailable_until_enabled(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = fixture.build_context()?;

    assert_eq!(
        context.nextn_embeddings_ith(0),
        Err(EmbeddingsError::NextnEmbeddingsNotEnabled)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    n_seq_max = 2,
)]
fn a_released_fork_leaves_its_shared_state_as_an_independent_decode_would(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let state = tokenize(fixture, STATE)?;
    let first_branch = tokenize(fixture, FIRST_BRANCH)?;
    let second_branch = tokenize(fixture, SECOND_BRANCH)?;
    let branch_position = i32::try_from(state.len())?;

    let mut context = hidden_state_context(fixture)?;
    let mut state_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(&mut state_batch, &state, 0, 0, false)?;
    context.decode(&mut state_batch)?;
    context.copy_kv_cache_seq(0, 1, None, None)?;

    let mut first_branch_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(
        &mut first_branch_batch,
        &first_branch,
        branch_position,
        1,
        true,
    )?;
    context.decode(&mut first_branch_batch)?;

    let first_branch_hidden_state = context
        .nextn_embeddings_ith(first_branch_batch.n_tokens() - 1)?
        .to_vec();

    context.clear_kv_cache_seq(Some(1), None, None)?;

    let mut second_branch_batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(
        &mut second_branch_batch,
        &second_branch,
        branch_position,
        0,
        true,
    )?;
    context.decode(&mut second_branch_batch)?;

    assert_eq!(
        first_branch_hidden_state,
        hidden_state_after(fixture, &state, &first_branch)?,
    );
    assert_eq!(
        context.nextn_embeddings_ith(second_branch_batch.n_tokens() - 1)?,
        hidden_state_after(fixture, &state, &second_branch)?.as_slice(),
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn tokenizing_without_special_token_parsing_keeps_control_markup_as_text(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let markup = "<|fim_prefix|>";
    let parsed = fixture
        .model
        .str_to_token(markup, AddBos::Never, ParseSpecialTokens::Always)?;
    let plain = fixture
        .model
        .str_to_token(markup, AddBos::Never, ParseSpecialTokens::Never)?;

    assert_eq!(parsed.len(), 1);
    assert!(plain.len() > 1);
    assert!(!plain.contains(&parsed[0]));

    Ok(())
}

fn decode_tokens(
    context: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    first_position: i32,
    output_on_last_token: bool,
) -> Result<i32> {
    let mut batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    add_tokens(&mut batch, tokens, first_position, 0, output_on_last_token)?;
    context.decode(&mut batch)?;

    Ok(batch.n_tokens() - 1)
}

fn final_hidden_state_context<'fixture>(
    fixture: &'fixture LlamaFixture<'_>,
    embeddings: bool,
) -> Result<LlamaContext<'fixture>> {
    Ok(LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_embeddings(embeddings)
            .with_pooling_type(LlamaPoolingType::None),
    )?)
}

fn row_hidden_state_after_enabling_embeddings(
    fixture: &LlamaFixture<'_>,
    state: &[LlamaToken],
    row: &[LlamaToken],
) -> Result<Vec<f32>> {
    let mut context = final_hidden_state_context(fixture, false)?;

    decode_tokens(&mut context, state, 0, false)?;
    context.set_embeddings(true);

    let row_last_index = decode_tokens(&mut context, row, i32::try_from(state.len())?, true)?;

    Ok(context.embeddings_ith(row_last_index)?.to_vec())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn final_hidden_states_after_enabling_embeddings_for_a_row_equal_those_of_an_embeddings_context(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let state = tokenize(fixture, STATE)?;
    let row = tokenize(fixture, FIRST_BRANCH)?;
    let toggled_row_hidden_state =
        row_hidden_state_after_enabling_embeddings(fixture, &state, &row)?;

    let mut embeddings_context = final_hidden_state_context(fixture, true)?;

    decode_tokens(&mut embeddings_context, &state, 0, false)?;

    let row_last_index = decode_tokens(
        &mut embeddings_context,
        &row,
        i32::try_from(state.len())?,
        true,
    )?;

    assert_eq!(
        toggled_row_hidden_state.as_slice(),
        embeddings_context.embeddings_ith(row_last_index)?,
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn disabling_embeddings_makes_token_embeddings_unavailable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let state = tokenize(fixture, STATE)?;
    let mut context = final_hidden_state_context(fixture, true)?;
    let last_index = decode_tokens(&mut context, &state, 0, true)?;

    context.set_embeddings(false);

    assert_eq!(
        context.embeddings_ith(last_index),
        Err(EmbeddingsError::NotEnabled)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn final_hidden_states_of_a_row_follow_the_state_decoded_before_it(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let row = tokenize(fixture, FIRST_BRANCH)?;

    assert_ne!(
        row_hidden_state_after_enabling_embeddings(fixture, &tokenize(fixture, STATE)?, &row)?,
        row_hidden_state_after_enabling_embeddings(
            fixture,
            &tokenize(fixture, OTHER_STATE)?,
            &row
        )?,
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn nextn_embeddings_are_refused_and_stay_disabled_on_a_model_without_nextn_hidden_states(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = final_hidden_state_context(fixture, false)?;

    assert_eq!(
        context.enable_masked_nextn_embeddings(),
        Err(EmbeddingsError::NextnEmbeddingsUnsupported)
    );
    assert_eq!(
        context.nextn_embeddings_ith(0),
        Err(EmbeddingsError::NextnEmbeddingsNotEnabled)
    );

    Ok(())
}
