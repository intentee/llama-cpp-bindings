use anyhow::Result;
use anyhow::ensure;

use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn encoding_with_a_decoder_only_model_reports_it_has_no_encoder(
    fixture @ &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&batch);

    ensure!(result == Err(ContextError::EncodeModelHasNoEncoder));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn an_encoder_model_encodes_a_prompt(
    fixture @ &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.encode(&batch);

    ensure!(result == Ok(()));

    Ok(())
}
