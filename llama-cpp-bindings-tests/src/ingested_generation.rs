use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;

/// Feeds the tokens of `generated_text` to `classifier` as if the model generated them, then
/// ends the generation with the model's end-of-sequence token.
///
/// # Panics
///
/// Panics when tokenizing or classifying fails, or when the classifier ends the generation
/// before the end-of-sequence token, or does not end it on that token.
pub fn ingested_generation(
    classifier: &mut SampledTokenClassifier<'_, '_>,
    model: &LlamaModel,
    generated_text: &str,
) -> Vec<IngestOutcome> {
    let mut outcomes = Vec::new();

    for token in model
        .str_to_token(generated_text, AddBos::Never, ParseSpecialTokens::Always)
        .expect("the generated text must tokenize")
    {
        assert_eq!(
            classifier.ingest(token, &mut outcomes),
            Ok(GenerationProgress::Continues)
        );
    }

    assert_eq!(
        classifier.ingest(
            model
                .token_eos()
                .expect("the model must declare an EOS token"),
            &mut outcomes
        ),
        Ok(GenerationProgress::Ended)
    );

    outcomes
}
