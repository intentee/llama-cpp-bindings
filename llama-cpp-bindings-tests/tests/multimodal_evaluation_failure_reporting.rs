use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_mtmd::evaluated_mtmd_input_chunk::EvaluatedMtmdInputChunk;
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
fn eval_chunks_reports_an_exception_llama_cpp_throws_while_evaluating_a_chunk(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_EVAL_CHUNK_SINGLE_THROWS,
    );

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::EvalReported {
                message: injected_fault_message()?
            })
    );

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
fn eval_each_yields_nothing_after_the_first_chunk_that_fails(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__> briefly")?;

    ensure!(chunks.len() > 2, "a chunk must follow the one that fails");

    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, n_batch, 0)?;
    let _fault = ArmedFault::after_passing(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_EVAL_CHUNK_SINGLE_THROWS,
        1,
    );
    let mut next_positions = fitted_chunks
        .eval_each(&mut llama_ctx, 0, 0, n_batch, true)?
        .map(|evaluated_chunk| {
            evaluated_chunk.map(|EvaluatedMtmdInputChunk { next_position, .. }| next_position)
        });

    ensure!(
        next_positions.nth(1)
            == Some(Err(MtmdError::EvalReported {
                message: injected_fault_message()?
            }))
    );
    ensure!(next_positions.next().is_none());

    Ok(())
}
