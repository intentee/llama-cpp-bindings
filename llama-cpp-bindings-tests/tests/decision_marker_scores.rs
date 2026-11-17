use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use serde_json::json;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings_tests::highest_score_index::highest_score_index;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const CHOICE_SCORE_COLUMN: usize = 0;

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn laya_scores_the_matching_option_highest_at_its_mask_token(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mask_token = model
        .token_mask()
        .context("a laya model marks its options with the mask token")?;
    let prompt = model.chat_template(Some("systemone"))?.render(
        &json!({
            "id": "weather",
            "type": "choice",
            "instructions": "What is the weather like?",
            "state": "Thick snowflakes have been falling on the frozen street all morning.",
            "options": [
                {"key": "sunny", "description": null},
                {"key": "raining", "description": null},
                {"key": "snowing", "description": null},
            ],
        })
        .to_string(),
    )?;
    let tokens = model.str_to_token(&prompt, AddBos::Never, ParseSpecialTokens::Always)?;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::None),
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&tokens, 0, true)?;
    context.decode(&batch)?;

    let mut option_scores = Vec::new();

    for (token_index, token) in (0..).zip(&tokens) {
        if *token == mask_token {
            option_scores.push(
                *context
                    .embeddings_ith(token_index)?
                    .get(CHOICE_SCORE_COLUMN)
                    .context("the token's embedding must hold the choice score column")?,
            );
        }
    }

    ensure!(option_scores.len() == 3, "one mask token marks each option");
    ensure!(
        highest_score_index(&option_scores) == Some(2),
        "option scores were {option_scores:?}"
    );

    Ok(())
}
