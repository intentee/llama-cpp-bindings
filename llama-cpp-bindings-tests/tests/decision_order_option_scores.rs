use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use serde_json::json;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::decision_order::DecisionOrder;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings_tests::highest_score_index::highest_score_index;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const PIECE_SEPARATOR: &str = "<<clef:sep>>";
const QUESTION_MARK: &str = "<<clef:question>>";
const OPTION_MARK: &str = "<<clef:option>>";
const OPTION_COUNT: i32 = 3;

fn marked_piece(piece: &str) -> MarkedPiece<'_> {
    piece.strip_prefix(QUESTION_MARK).map_or_else(
        || {
            piece.strip_prefix(OPTION_MARK).map_or(
                MarkedPiece {
                    decision_order: DecisionOrder::None,
                    text: piece,
                },
                |text| MarkedPiece {
                    decision_order: DecisionOrder::Option,
                    text,
                },
            )
        },
        |text| MarkedPiece {
            decision_order: DecisionOrder::QuestionChoice,
            text,
        },
    )
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Clef-Flash-GGUF", "Clef-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn clef_scores_the_matching_option_highest_among_the_options_marked_in_the_batch(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let prompt = model.chat_template(Some("systemone"))?.render(
        &json!({
            "state": "Thick snowflakes have been falling on the frozen street all morning.",
            "questions": [{
                "id": "weather",
                "type": "choice",
                "instructions": "What is the weather like?",
                "options": [
                    {"key": "raining", "description": null},
                    {"key": "snowing", "description": null},
                    {"key": "sunny", "description": null},
                ],
            }],
            "images": [],
            "sep": PIECE_SEPARATOR,
            "mark_question": QUESTION_MARK,
            "mark_option": OPTION_MARK,
        })
        .to_string(),
    )?;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::None),
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    for piece in prompt.split(PIECE_SEPARATOR) {
        let MarkedPiece {
            decision_order,
            text,
        } = marked_piece(piece);

        for token in model.str_to_token(text, AddBos::Never, ParseSpecialTokens::Always)? {
            let token_index = batch.n_tokens();

            batch.add(SampledToken::Content(token), token_index, &[0], true)?;
            batch.set_decision_order(token_index, decision_order)?;
        }
    }

    context.decode(&batch)?;

    let mut option_scores = Vec::new();

    for option_index in 0..OPTION_COUNT {
        option_scores.push(
            *context
                .embeddings_ith(option_index)?
                .first()
                .context("the option's embedding must hold its score")?,
        );
    }

    ensure!(
        highest_score_index(&option_scores) == Some(1),
        "option scores were {option_scores:?}"
    );

    Ok(())
}

struct MarkedPiece<'piece> {
    decision_order: DecisionOrder,
    text: &'piece str,
}
