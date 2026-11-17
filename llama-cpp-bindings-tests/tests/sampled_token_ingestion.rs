use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::llama_cpp_error::LlamaCppError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::token_piece::TokenPiece;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_model::sampled_token_section::SampledTokenSection;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
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
fn chat_session_classifier_starts_in_the_pending_section(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let classifier = SampledTokenClassifier::for_chat_session(model, session)?;

    ensure!(classifier.current_section() == SampledTokenSection::Pending);
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let [ordinary_token] =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("\"hello\" must be a single token");
    };

    let mut outcomes = Vec::new();
    classifier.ingest(ordinary_token, &mut outcomes)?;
    classifier.finish(&mut outcomes)?;

    ensure!(
        outcomes
            == [IngestOutcome {
                sampled_token: SampledToken::Undeterminable(ordinary_token),
                piece: TokenPiece::Visible("hello".to_owned()),
            }]
    );
    ensure!(classifier.usage().undeterminable == 1);
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
fn finishing_a_chat_answer_settles_its_pending_tokens(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let [ordinary_token] =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("\"hello\" must be a single token");
    };

    let mut outcomes = Vec::new();
    classifier.ingest(ordinary_token, &mut outcomes)?;
    classifier.finish(&mut outcomes)?;

    ensure!(
        outcomes
            == [IngestOutcome {
                sampled_token: SampledToken::Content(ordinary_token),
                piece: TokenPiece::Visible("hello".to_owned()),
            }]
    );
    ensure!(classifier.usage().content == 1);
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let [ordinary_token] =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("\"hello\" must be a single token");
    };
    let mut outcomes = Vec::new();

    classifier.ingest(ordinary_token, &mut outcomes)?;

    ensure!(
        classifier.ingest(
            model.token_eos().context("model declares an EOS token")?,
            &mut outcomes
        )? == GenerationProgress::Ended
    );

    ensure!(classifier.usage().undeterminable == 2);
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let token = LlamaToken::new(model.n_vocab());
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut outcomes = Vec::new();
    let expected = Err(LlamaCppError::Detokenize(
        ModelError::PieceTokenOutsideVocabulary {
            token,
            n_vocab: model.n_vocab(),
        },
    ));

    ensure!(SampledTokenClassifier::undeterminable(model).ingest(token, &mut outcomes) == expected);
    ensure!(
        SampledTokenClassifier::for_chat_session(model, session)?.ingest(token, &mut outcomes)
            == expected
    );
    ensure!(outcomes == []);

    Ok(())
}

fn failed_classification() -> Result<LlamaCppError> {
    Ok(LlamaCppError::Classify(ChatError::ParseReported {
        message: injected_fault_message()?,
    }))
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "Hello!", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let generated = model.str_to_token("Hi", AddBos::Never, ParseSpecialTokens::Never)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    ensure!(
        classifier.ingest(
            *generated.first().context("generated has no element 0")?,
            &mut Vec::new()
        ) == Err(failed_classification()?)
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "Hello!", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let end_of_sequence = model
        .token_eos()
        .context("the model declares an EOS token")?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    ensure!(classifier.ingest(end_of_sequence, &mut Vec::new()) == Err(failed_classification()?));

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
    let mut batch = LlamaBatch::new(context.limits());

    classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;
    context.decode(&batch)?;

    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    ensure!(
        classifier
            .sample(
                &mut LlamaSampler::greedy()?,
                &context,
                batch
                    .n_tokens()
                    .checked_sub(1)
                    .context("`batch.n_tokens() - 1` must not overflow")?,
                &mut Vec::new(),
            )
            .err()
            == Some(LlamaCppError::SampleIngest(Box::new(
                failed_classification()?
            )))
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
fn an_undeterminable_classifier_has_no_reading_of_the_answer(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut classifier = SampledTokenClassifier::undeterminable(model);

    ingested_generation(&mut classifier, model, "hello")?;

    ensure!(classifier.latest_reading().is_none());

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
fn a_chat_answer_classifier_parses_the_answer_it_read_last(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;

    ingested_generation(&mut classifier, model, "Hello there.")?;

    ensure!(
        classifier
            .latest_reading()
            .context("the classifier must have read the answer")?
            .parse()
            == Ok(&ChatMessageParseOutcome::Recognized(
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    ensure!(
        SampledTokenClassifier::for_chat_session(model, session).err()
            == Some(LlamaCppError::EmptyAnswerReading(
                ChatError::ParseReported {
                    message: injected_fault_message()?,
                }
            ))
    );

    Ok(())
}
