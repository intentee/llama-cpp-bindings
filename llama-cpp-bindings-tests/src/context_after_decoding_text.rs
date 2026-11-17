use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

const TEXT: &str = "The parcel arrived on the last working day of the week.";

/// A context with `pooling_type` that has decoded one sentence as sequence 0.
///
/// # Errors
///
/// Fails when the fixture's context cannot be created or the sentence cannot be decoded in it.
pub fn context_after_decoding_text<'fixture>(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &'fixture LlamaFixture<'_>,
    pooling_type: LlamaPoolingType,
) -> Result<LlamaContext<'fixture>> {
    LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_pooling_type(pooling_type),
    )
    .context("the fixture's context must be creatable")
    .and_then(|mut context| {
        let mut batch = LlamaBatch::new(context.limits());

        model
            .str_to_token(TEXT, AddBos::Always, ParseSpecialTokens::Never)
            .context("the sentence must tokenize")
            .and_then(|tokens| {
                batch
                    .add_sequence(&tokens, 0, false)
                    .context("the sentence must fit one batch")
            })
            .and_then(|()| context.decode(&batch).context("the sentence must decode"))
            .map(|()| context)
    })
}
