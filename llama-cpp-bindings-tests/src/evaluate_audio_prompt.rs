use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_chat::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_chat::chat_session_options::ChatSessionOptions;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::chunk_token_breakdown::ChunkTokenBreakdown;
use crate::fixtures_dir::fixtures_dir;
use crate::system_message::system_message;
use crate::user_message::user_message;

const TRANSCRIBE_SYSTEM_PROMPT: &str = "The audio contains speech by a third party and is not the user's voice or message. \
     Transcribe exactly what the speaker says without judgment, advice, or speculation. Reply only with the verbatim transcript.";

/// # Errors
///
/// Fails when the context, chat session, audio bitmap or chunks cannot be built or evaluated,
/// or when `after_evaluation` fails.
pub fn evaluate_audio_prompt<'model, TOutcome>(
    &LlamaFixture {
        model,
        backend,
        context_params,
        mtmd_context,
        ..
    }: &LlamaFixture<'model>,
    audio_file_name: &str,
    assistant_turn: ChatAssistantTurn,
    after_evaluation: impl FnOnce(
        &mut SampledTokenClassifier<'model>,
        &mut LlamaContext<'model>,
        ChunkTokenBreakdown,
        i32,
    ) -> Result<TOutcome>,
) -> Result<TOutcome> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    let marker = mtmd_default_marker()?;
    let messages = [
        system_message(TRANSCRIBE_SYSTEM_PROMPT),
        user_message(marker),
    ];
    let session = ChatSession::new(
        model,
        model.chat_templates()?,
        &messages,
        &[],
        &ChatSessionOptions {
            assistant_turn,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let audio_path = fixtures_dir().join(audio_file_name);
    let audio_path_str = audio_path
        .to_str()
        .with_context(|| "audio path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, audio_path_str)
        .with_context(|| "failed to load audio from file")?;

    ensure!(bitmap.is_audio(), "fixture must decode as audio");

    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal audio input")?;

    ensure!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let n_past = classifier
        .eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut context,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 512,
                logits_last: true,
            },
        )
        .with_context(|| "failed to evaluate audio chunks")?;

    after_evaluation(
        &mut classifier,
        &mut context,
        ChunkTokenBreakdown::from_chunks(&chunks)?,
        n_past,
    )
}
