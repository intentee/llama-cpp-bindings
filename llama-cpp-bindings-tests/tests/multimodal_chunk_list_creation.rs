use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

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
fn a_new_chunk_list_is_empty(&LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>) -> Result<()> {
    let chunks = MtmdInputChunks::new(mtmd_context.context("mmproj_file declared in attribute")?)?;

    ensure!(chunks.is_empty());
    ensure!(chunks.len() == 0);

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
fn a_new_chunk_list_holds_no_chunk_at_any_index(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = MtmdInputChunks::new(mtmd_context.context("mmproj_file declared in attribute")?)?;

    ensure!(chunks.get(0).is_none());
    ensure!(chunks.get(999).is_none());

    Ok(())
}
