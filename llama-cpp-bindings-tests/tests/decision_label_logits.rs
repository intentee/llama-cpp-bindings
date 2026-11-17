use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use serde_json::json;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings_tests::highest_score_index::highest_score_index;
use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const OPTION_LABELS: [&str; 3] = ["A", "B", "C"];

#[llama_test(
    model_source = HuggingFace("ggml-org/OpenJev-GGUF", "OpenJev-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn openjev_gives_the_label_of_the_matching_option_the_highest_logit(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
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
            "images": [],
        })
        .to_string(),
    )?;
    let tokens = model.str_to_token(&prompt, AddBos::Never, ParseSpecialTokens::Always)?;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let logits = context.get_logits_ith(
        batch
            .n_tokens()
            .checked_sub(1)
            .context("`batch.n_tokens() - 1` must not overflow")?,
    )?;
    let mut label_logits = Vec::new();

    for label in OPTION_LABELS {
        let label_token = single_token(model, label)?;
        let label_index = usize::try_from(label_token.0).context("token ids are not negative")?;

        label_logits.push(
            *logits
                .get(label_index)
                .context("the logits must cover the label token")?,
        );
    }

    ensure!(
        highest_score_index(&label_logits) == Some(2),
        "label logits were {label_logits:?}"
    );

    Ok(())
}
