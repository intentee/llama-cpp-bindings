use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::BareJsonToolCalls;
use llama_cpp_bindings::GenerationProgress;
use llama_cpp_bindings::IngestError;
use llama_cpp_bindings::SampledToken;
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::TokenOutsideVocabulary;
use llama_cpp_bindings::TokenPiece;
use llama_cpp_bindings::TokenToStringError;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token_section::SampledTokenSection;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn chat_session_classifier_starts_in_the_pending_section(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", "[]", false)?;
    let classifier = SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);

    assert_eq!(classifier.current_section(), SampledTokenSection::Pending);
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn finishing_a_chat_answer_settles_its_pending_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let session = user_turn_chat_session(model, "hello", "[]", false)?;
    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
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
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
    n_gpu_layers = 999,
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
    let session = user_turn_chat_session(model, "hello", "[]", false)?;
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
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect)
            .ingest(token, &mut outcomes),
        expected
    );
    assert_eq!(outcomes, []);

    Ok(())
}
