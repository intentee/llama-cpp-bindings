use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_bindings_tests::decode_decision_text::decode_decision_text;
use llama_cpp_bindings_tests::snowy_weather_decision_request::snowy_weather_decision_request;
use llama_cpp_bindings_types::decision_answer::DecisionAnswer;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_decision::decision_prompt_outputs::DecisionPromptOutputs;
use llama_cpp_decision::prepared_decision_request::PreparedDecisionRequest;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("ggml-org/Kev-0.8B-GGUF", "Kev-0.8B-BF16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_prompt_whose_final_batch_starts_at_its_first_read_position_is_answered(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let decision_model = DecisionModel::new(Arc::clone(model))?;
    let mut prepared =
        PreparedDecisionRequest::new(&decision_model, &snowy_weather_decision_request())?;
    let decision_context_params = decision_model
        .output()
        .context_params((*context_params).into_llama_context_params());

    for prompt in prepared.prompts(None)?.prompts {
        let first_read_position = prompt.first_read_position();

        ensure!(!prompt.must_be_decoded_at_once() && first_read_position > 0);

        let mut context = LlamaContext::from_model(model, backend, decision_context_params)?;

        decode_decision_text(&mut context, &prompt, 0..first_read_position, 0, false)?;

        let final_batch_outputs = decode_decision_text(
            &mut context,
            &prompt,
            first_read_position..prompt.tokens().len(),
            llama_pos::try_from(first_read_position)?,
            true,
        )?;
        let answers = prepared.read_answers(
            &prompt,
            &context,
            DecisionPromptOutputs {
                batch_indices: &final_batch_outputs,
                sequence_id: 0,
            },
        )?;

        ensure!(
            matches!(
                answers.get("weather"),
                Some(DecisionAnswer::Choice { choice, .. }) if choice == "snowing"
            ),
            "answers were {answers:?}"
        );
    }

    Ok(())
}
