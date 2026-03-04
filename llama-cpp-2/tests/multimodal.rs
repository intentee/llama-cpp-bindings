#![cfg(all(feature = "llm-tests", feature = "mtmd"))]
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::num::NonZeroU32;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;

const HF_REPO: &str = "unsloth/Qwen3.5-2B-GGUF";
const HF_MODEL: &str = "Qwen3.5-2B-Q4_K_M.gguf";
const HF_MMPROJ: &str = "mmproj-F16.gguf";

fn download_file(filename: &str) -> Result<std::path::PathBuf> {
    let path = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(HF_REPO.to_string())
        .get(filename)?;

    Ok(path)
}

fn create_test_image() -> Vec<u8> {
    let width: usize = 64;
    let height: usize = 64;
    let mut data = vec![0u8; width * height * 3];

    for y_pixel in 0..height {
        for x_pixel in 0..width {
            let offset = (y_pixel * width + x_pixel) * 3;

            if x_pixel < width / 2 {
                // Left half: red
                data[offset] = 255;
                data[offset + 1] = 0;
                data[offset + 2] = 0;
            } else {
                // Right half: blue
                data[offset] = 0;
                data[offset + 1] = 0;
                data[offset + 2] = 255;
            }
        }
    }

    data
}

#[test]
fn multimodal_vision_inference_produces_output() -> Result<()> {
    let model_path = download_file(HF_MODEL)?;
    let mmproj_path = download_file(HF_MMPROJ)?;

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let n_ctx = NonZeroU32::new(4096);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(n_ctx)
        .with_n_batch(512);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create llama context")?;

    let mtmd_params = MtmdContextParams::default();
    let mmproj_path_str = mmproj_path
        .to_str()
        .with_context(|| "mmproj path is not valid UTF-8")?;
    let mtmd_ctx = MtmdContext::init_from_file(mmproj_path_str, &model, &mtmd_params)
        .with_context(|| "unable to create mtmd context")?;

    assert!(
        mtmd_ctx.support_vision(),
        "model should support vision input"
    );

    let image_data = create_test_image();
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)
        .with_context(|| "failed to create bitmap from image data")?;

    let marker = llama_cpp_2::mtmd::mtmd_default_marker();
    let user_content = format!("{marker}What colors do you see in this image?");
    let chat_template = model.chat_template(None)?;
    let messages = [LlamaChatMessage::new("user".to_string(), user_content)?];
    let formatted_prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let input_text = MtmdInputText {
        text: formatted_prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal input")?;

    assert!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let total_tokens = chunks.total_tokens();
    eprintln!(
        "tokenized into {} chunks, {} total tokens",
        chunks.len(),
        total_tokens
    );

    let n_past = chunks
        .eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)
        .with_context(|| "failed to evaluate chunks")?;

    eprintln!("evaluated chunks, n_past = {n_past}");

    let mut sampler = LlamaSampler::greedy();
    let mut generated = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let max_tokens = 128;

    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);
    let mut current_position = n_past;

    for _ in 0..max_tokens {
        let token = sampler.sample(&ctx, -1);

        if model.is_eog_token(token) {
            break;
        }

        let output_string = model
            .token_to_piece(token, &mut decoder, false, None)
            .with_context(|| "failed to convert token to piece")?;
        generated.push_str(&output_string);

        batch.clear();
        batch.add(token, current_position, &[0], true)?;
        current_position += 1;

        ctx.decode(&mut batch)
            .with_context(|| "failed to decode generated token")?;
    }

    eprintln!("generated text: {generated}");

    assert!(
        !generated.is_empty(),
        "model should generate at least one token from image input"
    );

    Ok(())
}
