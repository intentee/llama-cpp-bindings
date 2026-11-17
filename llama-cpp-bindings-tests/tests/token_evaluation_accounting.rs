use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn fixture_context<'model>(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'model>,
) -> Result<LlamaContext<'model>> {
    Ok(LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?)
}

fn greeting_batch(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
    context: &LlamaContext<'_>,
) -> Result<LlamaBatch> {
    let tokens = model.str_to_token("hello there", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());

    batch.add_sequence(&tokens, 0, false)?;

    Ok(batch)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_decoded_batch_counts_every_token_llama_cpp_evaluated(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture_context(fixture)?;
    let batch = greeting_batch(fixture, &context)?;

    context.decode(&batch)?;

    ensure!(context.evaluated_tokens() == u64::from(batch.n_tokens().unsigned_abs()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn an_aborted_decode_counts_no_evaluated_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture_context(fixture)?;
    let batch = greeting_batch(fixture, &context)?;

    context.set_abort_flag(Arc::new(AtomicBool::new(true)));

    ensure!(context.decode(&batch).is_err());
    ensure!(context.evaluated_tokens() == 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn evaluated_media_chunks_count_every_token_the_mtmd_helper_decoded(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let bitmap = MtmdBitmap::from_image_data(64, 64, &[128_u8; 64 * 64 * 3])?;
    let chunks = MtmdInputChunks::tokenize(
        mtmd_ctx,
        MtmdInputText {
            text: "Describe: <__media__>".to_owned(),
            add_special: true,
            parse_special: true,
        },
        &[&bitmap],
    )?;
    let mut context = fixture_context(fixture)?;
    let n_batch = i32::try_from(context.n_batch())?;

    chunks.eval_chunks(mtmd_ctx, &mut context, 0, 0, n_batch, true)?;

    ensure!(context.evaluated_tokens() == chunks.total_tokens() as u64);

    Ok(())
}
