use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_capability::MtmdCapability;
use llama_cpp_mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const AUDIO_SAMPLE_RATE_HZ: u32 = 16_000;

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
fn qwen_image_chunks_decode_causally(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&bitmap])?;
    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .context(format!("missing chunk at index {chunk_index}"))?;
        if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
            ensure!(!chunk.decode_use_non_causal(mtmd_ctx));

            return Ok(());
        }
    }
    bail!("tokenization should produce at least one Image chunk");
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
fn qwen_mmproj_reports_vision_only_chat_capabilities(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    ensure!(
        mtmd_ctx.capabilities(model)
            == (MtmdCapability::Chat | MtmdCapability::Vision | MtmdCapability::TextGeneration)
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
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn decode_use_mrope_is_true_for_qwen_vision(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    ensure!(
        mtmd_ctx.decode_use_mrope(),
        "Qwen 3.5 / 3.6 mmproj uses mrope; decode_use_mrope must return true"
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
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn get_audio_sample_rate_is_none_for_vision_only_mmproj(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    ensure!(
        mtmd_ctx.get_audio_sample_rate().is_none(),
        "Qwen 3.5 / 3.6 mmproj has no audio; get_audio_sample_rate must return None"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn audio_mmproj_reports_audio_support_at_sixteen_kilohertz(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    ensure!(
        mtmd_ctx.capabilities(model).contains(MtmdCapability::Audio),
        "an audio mmproj must report audio support"
    );
    ensure!(mtmd_ctx.get_audio_sample_rate() == Some(AUDIO_SAMPLE_RATE_HZ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma4_mmproj_reports_chat_vision_audio_audio_stream_and_text_generation_capabilities(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    ensure!(
        mtmd_ctx.capabilities(model)
            == (MtmdCapability::Chat
                | MtmdCapability::Vision
                | MtmdCapability::Audio
                | MtmdCapability::AudioStream
                | MtmdCapability::TextGeneration)
    );

    Ok(())
}
