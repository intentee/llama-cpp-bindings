use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::mtmd_input_chunks_error::MtmdInputChunksError;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::mtmd::mtmd_tokenize_error::MtmdTokenizeError;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_has_text_type(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello world <__media__>");
    let first_chunk = chunks.get(0).context("missing first chunk")?;
    assert_eq!(first_chunk.chunk_type()?, MtmdInputChunkType::Text);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_returns_text_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello world <__media__>");
    let first_chunk = chunks.get(0).context("missing first chunk")?;
    let tokens = first_chunk
        .text_tokens_of(first_chunk.chunk_type()?)
        .context("the first chunk of a text prompt must carry tokens")?;

    assert_ne!(tokens, Vec::<LlamaToken>::new());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_n_tokens_is_positive(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello world <__media__>");
    let first_chunk = chunks.get(0).context("missing first chunk")?;
    assert!(first_chunk.n_tokens() > 0);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_n_positions_is_positive(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello world <__media__>");
    let first_chunk = chunks.get(0).context("missing first chunk")?;
    assert!(first_chunk.n_positions() > 0);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_id_returns_none(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello <__media__>");
    let first_chunk = chunks.get(0).context("missing first chunk")?;
    assert_eq!(first_chunk.chunk_type()?, MtmdInputChunkType::Text);
    assert!(first_chunk.id().is_none());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn image_chunk_returns_none_for_text_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello <__media__>");
    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .context(format!("missing chunk at index {chunk_index}"))?;
        if chunk.chunk_type()? == MtmdInputChunkType::Image {
            assert_eq!(chunk.text_tokens_of(MtmdInputChunkType::Image), None);

            return Ok(());
        }
    }
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn image_chunk_id_returns_some(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Hello <__media__>");
    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .context(format!("missing chunk at index {chunk_index}"))?;
        if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
            assert!(chunk.id().is_some());
            return Ok(());
        }
    }
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenizing_text_with_an_image_produces_chunks(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe this image: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    assert!(!chunks.is_empty());
    assert!(chunks.total_tokens() > 0);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenize_bitmap_count_mismatch_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let input_text = MtmdInputText {
        text: "No media markers here".to_string(),
        add_special: true,
        parse_special: true,
    };
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let result = mtmd_ctx.tokenize(input_text, &[&bitmap]);
    assert!(result.is_err());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenize_with_null_byte_in_text_returns_error(fixture: &LlamaFixture<'_>) {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let input_text = MtmdInputText {
        text: "text\0null".to_string(),
        add_special: true,
        parse_special: true,
    };
    let result = mtmd_ctx.tokenize(input_text, &[]);
    assert!(result.is_err());
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn tokenizing_reports_a_chunk_list_llama_cpp_cannot_allocate(fixture: &LlamaFixture<'_>) {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNKS_INIT_THROWS);

    assert_eq!(
        mtmd_ctx
            .tokenize(
                MtmdInputText {
                    text: "Hello".to_owned(),
                    add_special: true,
                    parse_special: true,
                },
                &[],
            )
            .err(),
        Some(MtmdTokenizeError::InputChunksError(
            MtmdInputChunksError::Reported {
                message: injected_fault_message(),
            }
        ))
    );
}
