use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::bare_json_tool_calls::BareJsonToolCalls;
use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::media_question_chat_session::media_question_chat_session;
use llama_cpp_bindings_tests::prompt_question::PROMPT_QUESTION;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma4_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    let model = fixture.model;
    let backend = fixture.backend;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let session = media_question_chat_session(model, PROMPT_QUESTION)?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let n_past = classifier.eval_multimodal_chunks(
        &chunks,
        mtmd_ctx,
        &mut context,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch: 512,
            logits_last: true,
        },
    )?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0)?,
        LlamaSampler::top_k(40)?,
        LlamaSampler::top_p(0.9, 1)?,
        LlamaSampler::min_p(0.05, 1)?,
        LlamaSampler::temp(0.7)?,
        LlamaSampler::dist(0x00C0_FFEE)?,
    ])?;

    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run();

    let usage = classifier.usage();

    if outcome.observed_reasoning == 0 {
        bail!(
            "Gemma 4 multimodal + thinking: classifier must emit at least one Reasoning token \
             when the prompt opens a `<|channel>thought` block; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        bail!(
            "Gemma 4 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "mmproj-F16.gguf"),
)]
fn mistral3_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 512;

    let model = fixture.model;
    let backend = fixture.backend;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker()?;
    let messages = [
        LlamaChatMessage::new(
            "system".to_owned(),
            "# HOW YOU SHOULD THINK AND ANSWER\n\n\
         First draft your thinking process (inner monologue) until you arrive at a response. \
         Format your response using Markdown, and use LaTeX for any mathematical equations. \
         Write both your thoughts and the response in the same language as the input.\n\n\
         Your thinking process must follow the template below:\
         [THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. \
         Be as casual and as long as you want until you are confident to generate the response \
         to the user.[/THINK]Here, provide a self-contained response."
                .to_owned(),
        )?,
        LlamaChatMessage::new("user".to_owned(), format!("{marker}{PROMPT_QUESTION}"))?,
    ];
    let session = ChatSession::new(
        model,
        &messages,
        "[]",
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
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

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let n_past = classifier.eval_multimodal_chunks(
        &chunks,
        mtmd_ctx,
        &mut context,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch: 512,
            logits_last: true,
        },
    )?;

    let mut sampler = LlamaSampler::greedy()?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run();

    let usage = classifier.usage();

    if outcome.observed_reasoning == 0 {
        bail!(
            "Mistral 3 multimodal + thinking: classifier must emit at least one Reasoning token \
             when the model opens a `[THINK]` block; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        bail!(
            "Mistral 3 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn qwen35_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    let model = fixture.model;
    let backend = fixture.backend;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let session = media_question_chat_session(model, PROMPT_QUESTION)?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let n_past = classifier.eval_multimodal_chunks(
        &chunks,
        mtmd_ctx,
        &mut context,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch: 512,
            logits_last: true,
        },
    )?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0)?,
        LlamaSampler::top_k(40)?,
        LlamaSampler::top_p(0.9, 1)?,
        LlamaSampler::min_p(0.05, 1)?,
        LlamaSampler::temp(0.7)?,
        LlamaSampler::dist(0x00C0_FFEE)?,
    ])?;

    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run();

    let usage = classifier.usage();

    if outcome.observed_reasoning == 0 {
        bail!(
            "Qwen 3.5 multimodal + thinking: classifier must emit at least one Reasoning token \
             when the prompt opens a `<think>` block; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        bail!(
            "Qwen 3.5 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn qwen36_classifier_emits_reasoning_for_multimodal_thinking_prompt(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const MAX_GENERATED_TOKENS: i32 = 200;

    let model = fixture.model;
    let backend = fixture.backend;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let session = media_question_chat_session(model, PROMPT_QUESTION)?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let n_past = classifier.eval_multimodal_chunks(
        &chunks,
        mtmd_ctx,
        &mut context,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch: 512,
            logits_last: true,
        },
    )?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.1, 0.0, 0.0)?,
        LlamaSampler::top_k(40)?,
        LlamaSampler::top_p(0.9, 1)?,
        LlamaSampler::min_p(0.05, 1)?,
        LlamaSampler::temp(0.7)?,
        LlamaSampler::dist(0x00C0_FFEE)?,
    ])?;

    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: MAX_GENERATED_TOKENS,
    }
    .run();

    let usage = classifier.usage();

    if outcome.observed_reasoning == 0 {
        bail!(
            "Qwen 3.6 multimodal + thinking: classifier must emit at least one Reasoning token; outcome={outcome:?}"
        );
    }
    if usage.reasoning_tokens == 0 {
        bail!(
            "Qwen 3.6 multimodal + thinking: usage.reasoning_tokens must be non-zero; usage={usage:?}"
        );
    }

    Ok(())
}
