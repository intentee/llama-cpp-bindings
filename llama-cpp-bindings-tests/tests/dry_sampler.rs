use std::ffi::CString;

use anyhow::Result;

use llama_cpp_bindings::error::sampling_error::SamplingError;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn building_a_dry_sampler_with_sequence_breakers_succeeds(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let breakers: Vec<&[u8]> = vec![b"\n", b"\t"];

    LlamaSampler::dry(fixture.model, 1.5, 2.0, 128, 2, &breakers)?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn dry_sampler_refuses_a_sequence_breaker_with_a_nul_byte(fixture: &LlamaFixture<'_>) {
    let breakers: Vec<&[u8]> = vec![b"hello\0world"];
    let result = LlamaSampler::dry(fixture.model, 1.5, 2.0, 128, 2, breakers);

    assert_eq!(
        result.err(),
        Some(SamplingError::DrySequenceBreakerContainsNul {
            index: 0,
            source: CString::new(&b"hello\0world"[..]).unwrap_err(),
        })
    );
}
