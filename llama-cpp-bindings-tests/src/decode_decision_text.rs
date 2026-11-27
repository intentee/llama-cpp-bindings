use std::ops::Range;

use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_decision::decision_prompt::DecisionPrompt;
use llama_cpp_model::sampled_token::SampledToken;

/// Decodes the `text` tokens of `prompt` in one batch on sequence 0.
///
/// Returns the batch indices of the outputs the answer is read from: every token when the prompt
/// is read from every token, otherwise the last one, or none unless `outputs`.
///
/// # Errors
///
/// Fails when `text` lies outside the prompt or the batch cannot be decoded.
pub fn decode_decision_text(
    context: &mut LlamaContext<'_>,
    prompt: &DecisionPrompt,
    text: Range<usize>,
    first_position: llama_pos,
    outputs: bool,
) -> Result<Vec<i32>> {
    let mut batch = LlamaBatch::new(context.limits());
    let last_index = text.end.saturating_sub(1);
    let mut output_indices = Vec::new();

    for ((token_index, batch_index), llama_position) in text.zip(0_i32..).zip(first_position..) {
        let token = *prompt
            .tokens()
            .get(token_index)
            .context("a prompt segment lies within the prompt")?;
        let is_output = outputs && (prompt.outputs_every_token() || token_index == last_index);

        batch.add_with_decision_order(
            SampledToken::Content(token),
            llama_position,
            &[0],
            is_output,
            prompt.decision_order(token_index),
        )?;

        if is_output {
            output_indices.push(batch_index);
        }
    }

    context.decode(&batch)?;

    Ok(output_indices)
}
