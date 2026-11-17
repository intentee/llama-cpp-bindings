use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

const SYNTHETIC_IMAGE_BYTES: usize = 64 * 64 * 3;

/// Tokenizes `prompt` with a uniformly grey 64x64 image as its media.
///
/// # Errors
///
/// Fails when the fixture declares no multimodal projector, or the image or prompt cannot be
/// tokenized.
pub fn tokenize_prompt_with_synthetic_image(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
    prompt: &str,
) -> Result<MtmdInputChunks> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;
    let image_data = vec![128u8; SYNTHETIC_IMAGE_BYTES];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)
        .context("a 64x64 RGB image must make a bitmap")?;
    let input_text = MtmdInputText {
        text: prompt.to_owned(),
        add_special: true,
        parse_special: true,
    };

    mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .context("the prompt and image must tokenize")
}
