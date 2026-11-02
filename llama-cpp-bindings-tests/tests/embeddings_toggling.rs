use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const STATE: &str = "Shoes arrived two weeks late and in the wrong size.";
const ROW: &str = " Which team should handle this ticket?";
const OTHER_STATE: &str = "The weather in Oslo is cold, dark and windy all week.";

fn tokenize(fixture: &LlamaFixture<'_>, text: &str) -> Result<Vec<LlamaToken>> {
    Ok(fixture
        .model
        .str_to_token(text, AddBos::Never, ParseSpecialTokens::Never)?)
}

fn add_tokens(
    batch: &mut LlamaBatch,
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

fn decode_tokens(
    context: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    first_position: i32,
    output_on_last_token: bool,
) -> Result<i32> {
    let mut batch = LlamaBatch::new(ContextLimits::of(context));

    add_tokens(&mut batch, tokens, first_position, 0, output_on_last_token)?;
    context.decode(&batch)?;

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
    let row = tokenize(fixture, ROW)?;
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
fn final_hidden_states_of_a_row_follow_the_state_decoded_before_it(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let row = tokenize(fixture, ROW)?;

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
