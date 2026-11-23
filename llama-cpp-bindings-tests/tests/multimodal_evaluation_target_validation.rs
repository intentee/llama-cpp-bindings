use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::freshly_loaded_model::freshly_loaded_model;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_rejects_a_sequence_the_context_does_not_have_instead_of_aborting(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let n_seq_max = llama_ctx.n_seq_max();
    let missing_sequence = i32::try_from(n_seq_max)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, missing_sequence, n_batch, true)
            == Err(MtmdError::SequenceIdOutOfRange {
                seq_id: missing_sequence,
                n_seq_max,
            })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_rejects_a_context_created_for_another_model_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let another_model = freshly_loaded_model(fixture)?;
    let mut llama_ctx = LlamaContext::from_model(
        &another_model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::ContextModelIsNotTheProjectorModel)
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}
