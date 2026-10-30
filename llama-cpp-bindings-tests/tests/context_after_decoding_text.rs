use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::params::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings_types::BatchCapacity;
use llama_cpp_test_harness::LlamaFixture;

const TEXT: &str = "The parcel arrived on the last working day of the week.";

pub fn context_after_decoding_text<'fixture>(
    fixture: &'fixture LlamaFixture<'_>,
    pooling_type: LlamaPoolingType,
) -> Result<LlamaContext<'fixture>> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(pooling_type),
    )?;
    let tokens = fixture
        .model
        .str_to_token(TEXT, AddBos::Always, ParseSpecialTokens::Never)?;
    let mut batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    Ok(context)
}
