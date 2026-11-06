use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::error::ingest_error::IngestError;
use llama_cpp_bindings::error::parse_chat_message_error::ParseChatMessageError;
use llama_cpp_bindings::error::sample_error::SampleError;
use llama_cpp_bindings::error::token_outside_vocabulary::TokenOutsideVocabulary;
use llama_cpp_bindings::error::token_to_string_error::TokenToStringError;
use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampled_token_section::SampledTokenSection;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings::token_piece::TokenPiece;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

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
fn chat_session_classifier_starts_in_the_pending_section(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], false)?;
    let classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)?;

    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
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
fn undeterminable_classifier_releases_each_token_with_its_visible_and_raw_piece(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let [ordinary_token] =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("\"hello\" must be a single token");
    };

    let mut outcomes = Vec::new();
    classifier.ingest(ordinary_token, &mut outcomes)?;
    classifier.finish(&mut outcomes)?;

    assert_eq!(
        outcomes,
        [IngestOutcome {
            sampled_token: SampledToken::Undeterminable(ordinary_token),
            piece: TokenPiece::Visible("hello".to_owned()),
        }]
    );
    assert_eq!(classifier.usage().undeterminable_tokens, 1);
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
fn finishing_a_chat_answer_settles_its_pending_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)?;
    let [ordinary_token] =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("\"hello\" must be a single token");
    };

    let mut outcomes = Vec::new();
    classifier.ingest(ordinary_token, &mut outcomes)?;
    classifier.finish(&mut outcomes)?;

    assert_eq!(
        outcomes,
        [IngestOutcome {
            sampled_token: SampledToken::Content(ordinary_token),
            piece: TokenPiece::Visible("hello".to_owned()),
        }]
    );
    assert_eq!(classifier.usage().content_tokens, 1);
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
fn ingest_counts_every_token_through_the_end_of_generation(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let [ordinary_token] =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("\"hello\" must be a single token");
    };
    let mut outcomes = Vec::new();

    classifier.ingest(ordinary_token, &mut outcomes)?;

    assert_eq!(
        classifier.ingest(
            model.token_eos().context("model declares an EOS token")?,
            &mut outcomes
        )?,
        GenerationProgress::Ended
    );

    assert_eq!(classifier.usage().undeterminable_tokens, 2);
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
fn ingesting_a_token_outside_the_vocabulary_reports_it_in_every_classifier(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let token = LlamaToken::new(model.n_vocab());
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut outcomes = Vec::new();
    let expected = Err(IngestError::Detokenize(
        TokenToStringError::TokenOutsideVocabulary(TokenOutsideVocabulary {
            token,
            n_vocab: model.n_vocab(),
        }),
    ));

    assert_eq!(
        SampledTokenClassifier::undeterminable(model).ingest(token, &mut outcomes),
        expected
    );
    assert_eq!(
        SampledTokenClassifier::for_chat_session(fixture.model, session)?
            .ingest(token, &mut outcomes),
        expected
    );
    assert_eq!(outcomes, []);

    Ok(())
}

fn failed_classification() -> IngestError {
    IngestError::Classify(ParseChatMessageError::Reported {
        message: injected_fault_message(),
    })
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn ingesting_a_generated_token_reports_a_classification_llama_cpp_fails(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "Hello!", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)?;
    let generated = fixture
        .model
        .str_to_token("Hi", AddBos::Never, ParseSpecialTokens::Never)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    assert_eq!(
        classifier.ingest(generated[0], &mut Vec::new()),
        Err(failed_classification())
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
fn ending_the_generation_reports_a_classification_llama_cpp_fails(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "Hello!", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)?;
    let end_of_sequence = fixture
        .model
        .token_eos()
        .context("the model declares an EOS token")?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    assert_eq!(
        classifier.ingest(end_of_sequence, &mut Vec::new()),
        Err(failed_classification())
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
fn sampling_reports_a_classification_llama_cpp_fails(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "Hello!", &[], false)?;
    let prompt_tokens = session.tokenize_prompt(fixture.model)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)?;
    let mut context = fixture.build_context()?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;
    context.decode(&batch)?;

    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    assert_eq!(
        classifier
            .sample(
                &mut LlamaSampler::greedy()?,
                &context,
                batch.n_tokens() - 1,
                &mut Vec::new(),
            )
            .err(),
        Some(SampleError::Ingest(failed_classification()))
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
fn an_undeterminable_classifier_has_no_reading_of_the_answer(fixture: &LlamaFixture<'_>) {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);

    ingested_generation(&mut classifier, model, "hello");

    assert!(classifier.latest_reading().is_none());
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_chat_answer_classifier_parses_the_answer_it_read_last(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)?;

    ingested_generation(&mut classifier, fixture.model, "Hello there.");

    assert_eq!(
        classifier
            .latest_reading()
            .context("the classifier must have read the answer")?
            .parse(),
        Ok(&ChatMessageParseOutcome::Recognized(
            ParsedChatMessage::new("Hello there.".to_owned(), String::new(), Vec::new())
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
)]
fn a_chat_answer_classifier_reports_a_failed_reading_of_its_empty_answer(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], false)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    assert_eq!(
        SampledTokenClassifier::for_chat_session(fixture.model, session).err(),
        Some(ParseChatMessageError::Reported {
            message: injected_fault_message(),
        })
    );

    Ok(())
}
