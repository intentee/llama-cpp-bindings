use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::tokenize_question_about_llamas::tokenize_question_about_llamas;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
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
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn fitting_no_chunks_leaves_nothing_to_evaluate(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let llama_ctx = fixture.build_context()?;
    let chunks = MtmdInputChunks::new(mtmd_ctx)?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, i32::try_from(llama_ctx.n_batch())?, 0)?;

    ensure!(fitted_chunks.fitted_chunks().is_empty());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 256,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn fitting_reports_a_chunk_type_the_bindings_do_not_recognize(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let llama_ctx = fixture.build_context()?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNK_TYPE_UNKNOWN,
    );

    ensure!(
        chunks.fit_in(&llama_ctx, 256, 0).err()
            == Some(MtmdError::EvalUnknownChunkType(Box::new(
                MtmdError::UnknownInputChunkType(
                    llama_cpp_bindings_sys::bindings::MTMD_INPUT_CHUNK_TYPE_COUNT
                )
            )))
    );

    Ok(())
}
