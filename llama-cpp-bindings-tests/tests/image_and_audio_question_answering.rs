use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::user_message::user_message;
use llama_cpp_chat::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_chat::chat_session_options::ChatSessionOptions;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const MAX_GENERATED_TOKENS: i32 = 512;
const DESCRIBE_INSTRUCTION: &str =
    "Describe the animal shown in the image, then write the exact words spoken in the audio.";

fn describe_image_and_audio_chat_session(model: &LlamaModel) -> Result<ChatSession> {
    let marker = mtmd_default_marker()?;
    let user_content = format!("Image: {marker}\nAudio: {marker}\n{DESCRIBE_INSTRUCTION}");
    let messages = [user_message(&user_content)];

    Ok(ChatSession::new(
        model,
        model.chat_templates()?,
        &messages,
        &[],
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: false,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?)
}

fn load_fixture_bitmap(mtmd_ctx: &MtmdContext, file_name: &str) -> Result<MtmdBitmap> {
    let path = fixtures_dir().join(file_name);
    let path_str = path
        .to_str()
        .context(format!("{file_name} path is not valid UTF-8"))?;
    MtmdBitmap::from_file(mtmd_ctx, path_str)
        .context(format!("failed to load {file_name} from file"))
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn an_image_and_audio_prompt_is_answered_with_both_media_counted_as_input(
    &LlamaFixture {
        model,
        backend,
        context_params,
        mtmd_context,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = mtmd_context.context("mmproj_file declared in attribute")?;

    let image_bitmap = load_fixture_bitmap(mtmd_ctx, "llamas.jpg")?;
    ensure!(!image_bitmap.is_audio(), "llamas.jpg must decode as image");

    let audio_bitmap = load_fixture_bitmap(mtmd_ctx, "orange_cat.wav")?;
    ensure!(
        audio_bitmap.is_audio(),
        "orange_cat.wav must decode as audio"
    );

    let session = describe_image_and_audio_chat_session(model)?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&image_bitmap, &audio_bitmap])
        .with_context(|| "failed to tokenize combined image and audio input")?;

    let expected = ChunkTokenBreakdown::from_chunks(&chunks)?;
    ensure!(
        expected.image > 0,
        "image input must produce at least one image-chunk token"
    );
    ensure!(
        expected.audio > 0,
        "audio input must produce at least one audio-chunk token"
    );

    let required_n_ctx = u32::try_from(
        chunks
            .total_positions()
            .checked_add(MAX_GENERATED_TOKENS)
            .context("`chunks.total_positions() + MAX_GENERATED_TOKENS` must not overflow")?,
    )?;
    ensure!(
        context_params.n_ctx >= required_n_ctx,
        "fixture n_ctx ({}) below required ({}); update the attribute literal",
        context_params.n_ctx,
        required_n_ctx,
    );

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    let n_batch = i32::try_from(context.n_batch())?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let n_past = classifier
        .eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut context,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch,
                logits_last: true,
            },
        )
        .with_context(|| "failed to evaluate image and audio chunks")?;

    {
        let usage = classifier.usage();
        ensure!(usage.input_image == expected.image);
        ensure!(usage.input_audio == expected.audio);
        ensure!(usage.prompt == expected.text);
    }

    let mut sampler = LlamaSampler::greedy()?;
    let mut batch = LlamaBatch::new(context.limits());
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run()?;

    ensure!(outcome.eog_seen, "{outcome:?}");
    ensure!(
        outcome.generated.text()
            == "Animal: The image shows a llama.\nAudio: The orange cat sat on the wooden fence."
    );

    Ok(())
}
