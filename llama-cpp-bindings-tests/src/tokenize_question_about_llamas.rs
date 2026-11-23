use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::fixtures_dir::fixtures_dir;
use crate::media_question_chat_session::media_question_chat_session;
use crate::prompt_question::PROMPT_QUESTION;

/// Tokenizes the prompt question about llamas, asked of the llamas photo.
///
/// # Errors
///
/// Fails when the llamas image cannot be loaded, or the question cannot be rendered or
/// tokenized.
pub fn tokenize_question_about_llamas<'context>(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
    mtmd_ctx: &'context MtmdContext,
) -> Result<MtmdInputChunks<'context>> {
    let image_path = fixtures_dir().join("llamas.jpg");

    image_path
        .to_str()
        .context("the llamas image path must be UTF-8")
        .and_then(|image_path| {
            MtmdBitmap::from_file(mtmd_ctx, image_path).context("the llamas image must load")
        })
        .and_then(|bitmap| {
            media_question_chat_session(model, PROMPT_QUESTION)
                .and_then(|session| session.prompt().context("the question must render"))
                .and_then(|text| {
                    MtmdInputChunks::tokenize(
                        mtmd_ctx,
                        MtmdInputText {
                            text,
                            add_special: true,
                            parse_special: true,
                        },
                        &[&bitmap],
                    )
                    .context("the question and image must tokenize")
                })
        })
}
