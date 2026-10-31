use anyhow::Result;
use llama_cpp_bindings::StringToTokenError;
use llama_cpp_bindings::batch_add_error::BatchAddError;
use llama_cpp_bindings_tests::prime_memory_with::prime_memory_with;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 64,
)]
fn prime_memory_surfaces_each_underlying_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    assert!(
        prime_memory_with(fixture, &mut context, "Hello\0world")
            .is_err_and(|error| error.downcast_ref::<StringToTokenError>().is_some()),
        "an interior null byte must surface a tokenization error"
    );
    assert_eq!(
        prime_memory_with(fixture, &mut context, &"word ".repeat(300))
            .err()
            .and_then(|error| error.downcast_ref::<BatchAddError>().cloned()),
        Some(BatchAddError::BatchFull { n_batch: 256 }),
        "more tokens than one decode takes must surface an add-sequence error"
    );

    let filler = "word ".repeat(40);
    let mut decode_result = prime_memory_with(fixture, &mut context, &filler);
    let mut attempts = 0;
    while decode_result.is_ok() && attempts < 16 {
        decode_result = prime_memory_with(fixture, &mut context, &filler);
        attempts += 1;
    }
    assert!(
        decode_result.is_err(),
        "filling the context past its window must surface a decode error"
    );

    Ok(())
}
