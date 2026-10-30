use anyhow::Result;
use llama_cpp_bindings::mtmd::MtmdBitmap;
use llama_cpp_bindings::mtmd::MtmdInputChunks;
use llama_cpp_bindings::mtmd::MtmdInputText;
use llama_cpp_test_harness::LlamaFixture;

pub fn tokenize_synthetic(fixture: &LlamaFixture<'_>, prompt: &str) -> Result<MtmdInputChunks> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; 64 * 64 * 3];
    let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data)?;
    let input_text = MtmdInputText {
        text: prompt.to_owned(),
        add_special: true,
        parse_special: true,
    };
    Ok(mtmd_ctx.tokenize(input_text, &[&bitmap])?)
}
