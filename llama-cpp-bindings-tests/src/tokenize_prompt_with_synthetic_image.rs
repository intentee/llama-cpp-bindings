use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// Tokenizes `prompt` with a uniformly grey 64x64 image as its media.
///
/// # Panics
///
/// Panics when the fixture declares no multimodal projector, or the image or prompt cannot be
/// tokenized.
#[must_use]
pub fn tokenize_prompt_with_synthetic_image(
    fixture: &LlamaFixture<'_>,
    prompt: &str,
) -> MtmdInputChunks {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)
        .expect("a 64x64 RGB image must make a bitmap");
    let input_text = MtmdInputText {
        text: prompt.to_owned(),
        add_special: true,
        parse_special: true,
    };

    mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .expect("the prompt and image must tokenize")
}
