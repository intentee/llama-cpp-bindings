use std::num::NonZeroU8;

use anyhow::Result;

use llama_cpp_bindings::error::memory_seq_add_error::MemorySeqAddError;
use llama_cpp_bindings::error::memory_seq_cp_error::MemorySeqCpError;
use llama_cpp_bindings::error::memory_seq_div_error::MemorySeqDivError;
use llama_cpp_bindings::error::memory_seq_pos_max_error::MemorySeqPosMaxError;
use llama_cpp_bindings::error::memory_seq_rm_error::MemorySeqRmError;
use llama_cpp_bindings::error::sequence_id_out_of_range::SequenceIdOutOfRange;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_pos_max_rejects_out_of_range_sequence(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = fixture.build_context()?;

    let result = context.memory_seq_pos_max(999);

    assert_eq!(
        result,
        Err(MemorySeqPosMaxError::SequenceIdOutOfRange(
            SequenceIdOutOfRange {
                seq_id: 999,
                n_seq_max: 1
            }
        ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_rm_rejects_sequence_id_exceeding_i32_max(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    let result = context.memory_seq_rm(Some(u32::MAX), None, None);

    assert_eq!(
        result.unwrap_err(),
        MemorySeqRmError::SeqIdTooLarge(
            i32::try_from(u32::MAX).expect_err("u32::MAX does not fit into i32")
        )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_cp_rejects_a_destination_outside_the_context_sequences(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    assert_eq!(
        context.memory_seq_cp(0, 1, None, None),
        Err(MemorySeqCpError::SequenceIdOutOfRange(
            SequenceIdOutOfRange {
                seq_id: 1,
                n_seq_max: 1
            }
        ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_keep_rejects_a_sequence_outside_the_context_sequences(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    assert_eq!(
        context.memory_seq_keep(1),
        Err(SequenceIdOutOfRange {
            seq_id: 1,
            n_seq_max: 1
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_add_rejects_a_negative_sequence_id(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = fixture.build_context()?;

    assert_eq!(
        context.memory_seq_add(-1, None, None, 1),
        Err(MemorySeqAddError::SequenceIdOutOfRange(
            SequenceIdOutOfRange {
                seq_id: -1,
                n_seq_max: 1
            }
        ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_div_rejects_a_sequence_outside_the_context_sequences(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    assert_eq!(
        context.memory_seq_div(1, None, None, NonZeroU8::MIN),
        Err(MemorySeqDivError::SequenceIdOutOfRange(
            SequenceIdOutOfRange {
                seq_id: 1,
                n_seq_max: 1
            }
        ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn memory_seq_rm_rejects_a_sequence_outside_the_context_sequences(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;

    assert_eq!(
        context.memory_seq_rm(Some(1), None, None),
        Err(MemorySeqRmError::SequenceIdOutOfRange(
            SequenceIdOutOfRange {
                seq_id: 1,
                n_seq_max: 1
            }
        ))
    );

    Ok(())
}
