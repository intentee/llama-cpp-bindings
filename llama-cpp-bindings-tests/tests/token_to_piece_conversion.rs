use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use std::ffi::c_int;
use std::num::NonZeroU16;

use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const PIECE_PROBE_BYTES: usize = 32;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_to_piece_bytes_of_the_hello_token_is_hello(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let tokens = model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;

    ensure!(
        model.token_to_piece_bytes(
            *tokens.first().context("tokens has no element 0")?,
            PIECE_PROBE_BYTES,
            false,
            None
        )? == b"hello"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_to_piece_returns_a_piece_longer_than_its_probe_buffer_whole(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let long_token = (0..model.n_vocab())
        .map(LlamaToken::new)
        .find(|token| {
            matches!(
                model.token_to_piece_bytes(*token, PIECE_PROBE_BYTES, false, None),
                Err(ModelError::PieceInsufficientBufferSpace(_))
            )
        })
        .context("the vocabulary has a piece longer than the probe buffer")?;
    let Err(ModelError::PieceInsufficientBufferSpace(negated_piece_size)) =
        model.token_to_piece_bytes(long_token, PIECE_PROBE_BYTES, false, None)
    else {
        bail!("the probe buffer must be too small for the long piece");
    };
    let long_piece = model.token_to_piece_bytes(
        long_token,
        usize::try_from(negated_piece_size.unsigned_abs())?,
        false,
        None,
    )?;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    ensure!(
        model.token_to_piece(SampledToken::Content(long_token), &mut decoder, false, None)?
            == String::from_utf8(long_piece)?
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_to_piece_bytes_with_a_too_small_buffer_reports_the_size_the_piece_needs(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let tokens = model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;

    ensure!(
        model.token_to_piece_bytes(
            *tokens.first().context("tokens has no element 0")?,
            1,
            false,
            None
        ) == Err(ModelError::PieceInsufficientBufferSpace(
            c_int::try_from("hello".len())?
                .checked_neg()
                .context("`-c_int::try_from(\"hello\".len())?` must not overflow")?
        ))
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_to_piece_strips_the_requested_count_of_leading_spaces(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let tokens = model.str_to_token(" hello", AddBos::Never, ParseSpecialTokens::Always)?;

    ensure!(
        model.token_to_piece(
            SampledToken::Content(*tokens.first().context("tokens has no element 0")?),
            &mut decoder,
            false,
            NonZeroU16::new(1),
        )? == "hello"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_to_piece_decodes_a_token_alike_in_every_non_content_section(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let hi_token = single_token(model, "hi")?;

    ensure!(
        [
            SampledToken::Reasoning(hi_token),
            SampledToken::ToolCall(hi_token),
            SampledToken::Undeterminable(hi_token),
        ]
        .map(|sampled_token| {
            model.token_to_piece(
                sampled_token,
                &mut encoding_rs::UTF_8.new_decoder(),
                true,
                None,
            )
        }) == [
            Ok("hi".to_owned()),
            Ok("hi".to_owned()),
            Ok("hi".to_owned())
        ]
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
)]
fn token_to_piece_bytes_refuses_a_buffer_beyond_a_c_int_before_allocating(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        model
            .token_to_piece_bytes(LlamaToken::new(0), usize::MAX, false, None)
            .err(),
        i32::try_from(usize::MAX)
            .err()
            .map(ModelError::PieceIntConversion)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_to_piece_replaces_a_lone_continuation_byte_with_the_replacement_character(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let continuation_byte_token = (0..model.n_vocab())
        .map(LlamaToken::new)
        .find(|token| {
            model
                .token_to_piece_bytes(*token, 8, true, None)
                .is_ok_and(|bytes| bytes == [0x80])
        })
        .context("the byte-level vocabulary has a token for the byte 0x80")?;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    ensure!(
        model.token_to_piece(
            SampledToken::Content(continuation_byte_token),
            &mut decoder,
            true,
            None
        )? == "\u{FFFD}"
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
)]
fn a_control_token_decoded_without_special_rendering_is_an_empty_piece(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let control_token = single_token(model, "<|im_start|>")?;

    ensure!(
        model.token_to_piece(
            SampledToken::Content(control_token),
            &mut encoding_rs::UTF_8.new_decoder(),
            false,
            None
        ) == Ok(String::new())
    );

    Ok(())
}
