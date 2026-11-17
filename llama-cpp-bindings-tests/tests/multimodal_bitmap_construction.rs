use std::ffi::CString;
use std::fs;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_bitmap_error::MtmdBitmapError;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const LLAMAS_JPG_WIDTH: u32 = 640;
const LLAMAS_JPG_HEIGHT: u32 = 427;

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
fn from_buffer_decodes_image_bytes_into_an_image_bitmap_of_their_size(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    let fixtures = fixtures_dir();
    let image_path = fixtures.join("llamas.jpg");
    let image_bytes = fs::read(&image_path)?;
    let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, &image_bytes)?;

    ensure!(bitmap.nx() == LLAMAS_JPG_WIDTH);
    ensure!(bitmap.ny() == LLAMAS_JPG_HEIGHT);
    ensure!(!bitmap.is_audio());

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
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn from_file_refuses_a_path_with_a_nul_byte(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    ensure!(
        MtmdBitmap::from_file(mtmd_ctx, "path\0null").err()
            == CString::new("path\0null")
                .err()
                .map(MtmdBitmapError::CStringError)
    );

    Ok(())
}
