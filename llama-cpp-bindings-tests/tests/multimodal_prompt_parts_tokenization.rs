use std::ffi::CString;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_bindings::mtmd::mtmd_input_chunks_error::MtmdInputChunksError;
use llama_cpp_bindings::mtmd::mtmd_input_part::MtmdInputPart;
use llama_cpp_bindings::mtmd::mtmd_input_text_part::MtmdInputTextPart;
use llama_cpp_bindings::mtmd::mtmd_tokenize_error::MtmdTokenizeError;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

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
            .context(format!("missing chunk at index {chunk_index}"))?;

        if let Some(chunk_tokens) = chunk.text_tokens_of(chunk.chunk_type()?) {
            tokens.extend_from_slice(chunk_tokens);
        }
    }

    Ok(tokens)
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
fn only_bitmap_parts_become_image_chunks(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
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
    let mut image_chunks = 0_u32;

    for chunk_index in 0..chunks.len() {
        let chunk = chunks
            .get(chunk_index)
            .context(format!("missing chunk at index {chunk_index}"))?;

        if chunk.chunk_type()? == MtmdInputChunkType::Image {
            image_chunks = image_chunks
                .checked_add(1)
                .context("`image_chunks += 1` must not overflow")?;
        }
    }

    ensure!(image_chunks == 1);

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
fn a_media_marker_inside_a_text_part_tokenizes_as_plain_text(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let marker = mtmd_default_marker()?;
    let chunks = mtmd_ctx.tokenize_from_parts(&[text_part(marker, true)], false)?;

    ensure!(
        tokens_of_text_chunks(&chunks)?
            == model.str_to_token(marker, AddBos::Never, ParseSpecialTokens::Always)?
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
fn parse_special_applies_to_each_text_part_on_its_own(
    &LlamaFixture {
        model,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let chunks = mtmd_ctx.tokenize_from_parts(
        &[
            text_part(SPECIAL_TOKEN_TEXT, true),
            text_part(SPECIAL_TOKEN_TEXT, false),
        ],
        false,
    )?;

    ensure!(
        tokens_of_text_chunks(&chunks)?
            == [
                model.str_to_token(
                    SPECIAL_TOKEN_TEXT,
                    AddBos::Never,
                    ParseSpecialTokens::Always
                )?,
                model.str_to_token(SPECIAL_TOKEN_TEXT, AddBos::Never, ParseSpecialTokens::Never)?,
            ]
            .concat()
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
fn a_text_part_containing_a_nul_byte_is_rejected(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let text_with_nul_byte = "before\0after";

    ensure!(
        mtmd_ctx
            .tokenize_from_parts(&[text_part(text_with_nul_byte, true)], false)
            .err()
            == CString::new(text_with_nul_byte)
                .err()
                .map(MtmdTokenizeError::CStringError)
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
fn tokenizing_parts_reports_a_chunk_list_llama_cpp_cannot_allocate(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNKS_INIT_THROWS);

    ensure!(
        mtmd_ctx
            .tokenize_from_parts(&[text_part("Hello", true)], true)
            .err()
            == Some(MtmdTokenizeError::InputChunksError(
                MtmdInputChunksError::Reported {
                    message: injected_fault_message()?,
                }
            ))
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
fn tokenizing_parts_reports_a_tokenizer_llama_cpp_fails(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_TOKENIZE_FROM_PARTS_THROWS,
    );

    ensure!(
        mtmd_ctx
            .tokenize_from_parts(&[text_part("Hello", true)], true)
            .err()
            == Some(MtmdTokenizeError::Reported {
                message: injected_fault_message()?,
            })
    );

    Ok(())
}
