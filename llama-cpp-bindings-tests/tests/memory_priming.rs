use std::ffi::CString;

use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::batch_add_error::BatchAddError;
use llama_cpp_bindings::error::decode_error::DecodeError;
use llama_cpp_bindings::error::string_to_token_error::StringToTokenError;
use llama_cpp_bindings_tests::prime_memory::prime_memory;
use llama_cpp_bindings_tests::prime_memory_with::prime_memory_with;
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

    assert_eq!(
        Some(failure.downcast::<StringToTokenError>()?),
        CString::new("Hello\0world")
            .err()
            .map(StringToTokenError::NulError)
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

    assert_eq!(
        prime_memory_with(fixture, &mut context, &"word ".repeat(300))
            .err()
            .and_then(|error| error.downcast_ref::<BatchAddError>().cloned()),
        Some(BatchAddError::BatchFull { n_batch: 256 }),
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

    assert_eq!(
        failure.downcast::<DecodeError>()?,
        DecodeError::BatchInvalid
    );

    Ok(())
}
