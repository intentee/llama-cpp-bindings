use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use std::ffi::CString;

use llama_cpp_bindings_tests::prime_memory::prime_memory;
use llama_cpp_bindings_tests::prime_memory_with::prime_memory_with;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 64,
)]
fn priming_memory_surfaces_a_tokenization_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;
    let Err(failure) = prime_memory_with(fixture, &mut context, "Hello\0world") else {
        bail!("an interior NUL byte must refuse the tokenization");
    };

    ensure!(
        Some(failure.downcast::<ModelError>()?)
            == CString::new("Hello\0world")
                .err()
                .map(ModelError::TokenizeTextContainsNulByte)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 64,
)]
fn priming_memory_surfaces_a_batch_add_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    ensure!(
        prime_memory_with(fixture, &mut context, &"word ".repeat(300))
            .err()
            .and_then(|error| error.downcast_ref::<ContextError>().cloned())
            == Some(ContextError::BatchFull { n_batch: 256 }),
        "more tokens than one decode takes must surface an add-sequence error"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 64,
)]
fn priming_memory_surfaces_a_decode_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let Err(failure) = prime_memory(fixture, &mut context) else {
        bail!("decoding the primed sequence again from its first position must fail");
    };

    ensure!(failure.downcast::<ContextError>()? == ContextError::DecodeBatchInvalid);

    Ok(())
}
