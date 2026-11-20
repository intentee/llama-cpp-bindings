use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
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
fn a_logits_row_holds_one_logit_per_vocabulary_token(
    fixture @ &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let last_index = i32::try_from(
        tokens
            .len()
            .checked_sub(1)
            .context("`tokens.len() - 1` must not overflow")?,
    )?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let logits = context.get_logits_ith(last_index)?;

    ensure!(logits.len() == usize::try_from(model.n_vocab())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn logits_of_a_token_no_decode_initialized_are_refused(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = fixture.build_context()?;

    let result = context.get_logits_ith(7);

    ensure!(result == Err(ContextError::LogitTokenNotInitialized(7)));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn reading_a_logits_row_reports_a_row_llama_cpp_does_not_return(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_GET_LOGITS_ITH_RETURNS_NULL,
    );

    ensure!(context.get_logits_ith(-1).err() == Some(ContextError::LogitsNull));

    Ok(())
}
