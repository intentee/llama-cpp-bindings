use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use std::num::NonZeroU8;

use llama_cpp_bindings::error::memory_seq_add_error::MemorySeqAddError;
use llama_cpp_bindings::error::memory_seq_div_error::MemorySeqDivError;
use llama_cpp_bindings_tests::prime_memory::prime_memory;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_add_refuses_an_mrope_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let result = context.memory_seq_add(0, Some(0), None, 1);

    ensure!(
        result.err().context("the operation must fail")? == MemorySeqAddError::IncompatibleRopeType
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_div_refuses_an_mrope_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;
    let result = context.memory_seq_div(0, Some(0), None, divisor);

    ensure!(
        result.err().context("the operation must fail")? == MemorySeqDivError::IncompatibleRopeType
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_add_shifts_the_positions_of_a_standard_rope_model(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let primed_position_max = context.memory_seq_pos_max(0)?;

    context.memory_seq_add(0, Some(0), None, 1)?;

    ensure!(
        context.memory_seq_pos_max(0)?
            == primed_position_max
                .checked_add(1)
                .context("`primed_position_max + 1` must not overflow")?
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_div_divides_the_positions_of_a_standard_rope_model(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    prime_memory(fixture, &mut context)?;

    let primed_position_max = context.memory_seq_pos_max(0)?;
    let divisor = NonZeroU8::new(2).context("2 is non-zero")?;

    context.memory_seq_div(0, Some(0), None, divisor)?;

    ensure!(
        context.memory_seq_pos_max(0)?
            == primed_position_max
                .checked_div(i32::from(divisor.get()))
                .context("`primed_position_max / i32::from(divisor.get())` must not overflow")?
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_add_refuses_a_model_whose_rope_type_llama_cpp_does_not_define(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_ROPE_TYPE_UNKNOWN);

    ensure!(
        context.memory_seq_add(0, None, None, 1)
            == Err(MemorySeqAddError::UnknownRopeType(
                ModelError::UnknownRopeType(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_IMROPE + 1)
            ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_div_refuses_a_model_whose_rope_type_llama_cpp_does_not_define(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_ROPE_TYPE_UNKNOWN);

    ensure!(
        context.memory_seq_div(0, None, None, NonZeroU8::MIN)
            == Err(MemorySeqDivError::UnknownRopeType(
                ModelError::UnknownRopeType(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_IMROPE + 1)
            ))
    );

    Ok(())
}
