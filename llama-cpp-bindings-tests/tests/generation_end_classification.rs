use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_tests::visible_text_in::visible_text_in;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token_section::SampledTokenSection;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn visible_text(outcomes: &[IngestOutcome]) -> String {
    outcomes
        .iter()
        .map(|IngestOutcome { piece, .. }| piece.visible())
        .collect()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_classifier_ends_generation_without_emitting_the_end_of_generation_token(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    const GENERATED_TEXT: &str = "The answer is four.";

    let session = user_turn_chat_session(model, "What is 2 + 2?", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;

    ensure!(
        visible_text(&ingested_generation(
            &mut classifier,
            model,
            GENERATED_TEXT
        )?) == GENERATED_TEXT
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_answer_stopped_inside_unclosed_reasoning_releases_its_unsettled_tail_as_reasoning(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    const SETTLED_REASONING: &str = "Let me think";
    const UNSETTLED_TAIL: &str = "</thi";

    let session = user_turn_chat_session(model, "What is 2 + 2?", &[], true)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let mut outcomes = Vec::new();

    for token in model.str_to_token(
        &format!("{SETTLED_REASONING}{UNSETTLED_TAIL}"),
        AddBos::Never,
        ParseSpecialTokens::Always,
    )? {
        classifier.ingest(token, &mut outcomes)?;
    }
    classifier.finish(&mut outcomes)?;

    ensure!(
        [
            visible_text_in(&outcomes, SampledTokenSection::Reasoning),
            visible_text_in(&outcomes, SampledTokenSection::Content),
            visible_text_in(&outcomes, SampledTokenSection::ToolCall),
            visible_text_in(&outcomes, SampledTokenSection::Pending),
        ] == [
            format!("{SETTLED_REASONING}{UNSETTLED_TAIL}"),
            String::new(),
            String::new(),
            String::new()
        ]
    );

    Ok(())
}
