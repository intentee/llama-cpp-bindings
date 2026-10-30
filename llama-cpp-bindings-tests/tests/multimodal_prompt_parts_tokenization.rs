use std::ffi::CString;

use anyhow::Result;
use anyhow::anyhow;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::MtmdInputChunks;
use llama_cpp_bindings::mtmd::MtmdInputChunksError;
use llama_cpp_bindings::mtmd::MtmdInputPart;
use llama_cpp_bindings::mtmd::MtmdInputTextPart;
use llama_cpp_bindings::mtmd::MtmdTokenizeError;
use llama_cpp_bindings::mtmd::mtmd_default_marker;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const SPECIAL_TOKEN_TEXT: &str = "<|im_start|>";

fn text_part(text: &str, parse_special: bool) -> MtmdInputPart<'static> {
    MtmdInputPart::Text(MtmdInputTextPart {
        text: text.to_owned(),
        parse_special,
    })
}

fn tokens_of_text_chunks(chunks: &MtmdInputChunks) -> Result<Vec<LlamaToken>> {
    let mut tokens = Vec::new();

    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .ok_or_else(|| anyhow!("missing chunk at index {chunk_index}"))?;

        if let Some(chunk_tokens) = chunk.text_tokens_of(chunk.chunk_type()?) {
            tokens.extend_from_slice(chunk_tokens);
        }
    }

    Ok(tokens)
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
fn only_bitmap_parts_become_image_chunks(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let chunks = mtmd_ctx.tokenize_from_parts(
        &[
            text_part(
                &format!("What does {} mean? ", mtmd_default_marker()?),
                true,
            ),
            MtmdInputPart::Bitmap(&bitmap),
            text_part(" Describe it.", true),
        ],
        true,
    )?;
    let mut image_chunks = 0;

    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .ok_or_else(|| anyhow!("missing chunk at index {chunk_index}"))?;

        if chunk.chunk_type()? == MtmdInputChunkType::Image {
            image_chunks += 1;
        }
    }

    assert_eq!(image_chunks, 1);

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
fn a_media_marker_inside_a_text_part_tokenizes_as_plain_text(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let marker = mtmd_default_marker()?;
    let chunks = mtmd_ctx.tokenize_from_parts(&[text_part(marker, true)], false)?;

    assert_eq!(
        tokens_of_text_chunks(&chunks)?,
        fixture
            .model
            .str_to_token(marker, AddBos::Never, ParseSpecialTokens::Always)?
    );

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
fn parse_special_applies_to_each_text_part_on_its_own(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let chunks = mtmd_ctx.tokenize_from_parts(
        &[
            text_part(SPECIAL_TOKEN_TEXT, true),
            text_part(SPECIAL_TOKEN_TEXT, false),
        ],
        false,
    )?;

    assert_eq!(
        tokens_of_text_chunks(&chunks)?,
        [
            fixture.model.str_to_token(
                SPECIAL_TOKEN_TEXT,
                AddBos::Never,
                ParseSpecialTokens::Always
            )?,
            fixture.model.str_to_token(
                SPECIAL_TOKEN_TEXT,
                AddBos::Never,
                ParseSpecialTokens::Never
            )?,
        ]
        .concat()
    );

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
fn a_text_part_containing_a_nul_byte_is_rejected(fixture: &LlamaFixture<'_>) {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let text_with_nul_byte = "before\0after";

    assert_eq!(
        mtmd_ctx
            .tokenize_from_parts(&[text_part(text_with_nul_byte, true)], false)
            .err(),
        CString::new(text_with_nul_byte)
            .err()
            .map(MtmdTokenizeError::CStringError)
    );
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
fn tokenizing_parts_reports_a_chunk_list_llama_cpp_cannot_allocate(fixture: &LlamaFixture<'_>) {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNKS_INIT_THROWS);

    assert_eq!(
        mtmd_ctx
            .tokenize_from_parts(&[text_part("Hello", true)], true)
            .err(),
        Some(MtmdTokenizeError::InputChunksError(
            MtmdInputChunksError::Reported {
                message: injected_fault_message(),
            }
        ))
    );
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
fn tokenizing_parts_reports_a_tokenizer_llama_cpp_fails(fixture: &LlamaFixture<'_>) {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_TOKENIZE_FROM_PARTS_THROWS,
    );

    assert_eq!(
        mtmd_ctx
            .tokenize_from_parts(&[text_part("Hello", true)], true)
            .err(),
        Some(MtmdTokenizeError::Reported {
            message: injected_fault_message(),
        })
    );
}
