use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_test_harness::LlamaFixture;

const TEXT: &str = "The parcel arrived on the last working day of the week.";

/// A context with `pooling_type` that has decoded one sentence as sequence 0.
///
/// # Panics
///
/// Panics when the fixture's context cannot be created or the sentence cannot be decoded in it.
#[must_use]
pub fn context_after_decoding_text<'fixture>(
    fixture: &'fixture LlamaFixture<'_>,
    pooling_type: LlamaPoolingType,
) -> LlamaContext<'fixture> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(pooling_type),
    )
    .expect("the fixture's context must be creatable");
    let tokens = fixture
        .model
        .str_to_token(TEXT, AddBos::Always, ParseSpecialTokens::Never)
        .expect("the sentence must tokenize");
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch
        .add_sequence(&tokens, 0, false)
        .expect("the sentence must fit one batch");
    context.decode(&batch).expect("the sentence must decode");

    context
}
