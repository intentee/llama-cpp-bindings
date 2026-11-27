use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use anyhow::bail;
use indexmap::IndexMap;
use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_bindings_types::decision_answer::DecisionAnswer;
use llama_cpp_bindings_types::decision_request::DecisionRequest;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_decision::decision_prompt::DecisionPrompt;
use llama_cpp_decision::decision_prompt_outputs::DecisionPromptOutputs;
use llama_cpp_decision::prepared_decision_request::PreparedDecisionRequest;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_input_chunk_view::MtmdInputChunkView;
use llama_cpp_params::llama_context_params::LlamaContextParams;

use crate::decode_decision_text::decode_decision_text;

enum PromptSegment<'prompt, 'projector> {
    Media(MtmdInputChunkView<'prompt, 'projector>),
    Text(Range<usize>),
}

fn segments_of<'prompt, 'projector>(
    prompt: &'prompt DecisionPrompt,
    projector: Option<&'projector MtmdContext>,
) -> Vec<PromptSegment<'prompt, 'projector>> {
    let token_count = prompt.tokens().len();
    let mut segments = Vec::new();
    let mut text_start = 0;
    let mut position = 0;

    while position < token_count {
        match projector.and_then(|projector| prompt.media_chunk(position, projector)) {
            Some(chunk) => {
                if text_start < position {
                    segments.push(PromptSegment::Text(text_start..position));
                }
                segments.push(PromptSegment::Media(chunk));
                position = position.saturating_add(chunk.n_tokens());
                text_start = position;
            }
            None => position = position.saturating_add(1),
        }
    }

    if text_start < token_count {
        segments.push(PromptSegment::Text(text_start..token_count));
    }

    segments
}

fn answers_of_prompt_decoded_alone(
    prepared: &mut PreparedDecisionRequest,
    model: &LlamaModel,
    backend: &LlamaBackend,
    context_params: LlamaContextParams,
    prompt: &DecisionPrompt,
    projector: Option<&MtmdContext>,
) -> Result<IndexMap<String, DecisionAnswer>> {
    let mut context = LlamaContext::from_model(model, backend, context_params)?;
    let n_batch = i32::try_from(context.n_batch())?;
    let segments = segments_of(prompt, projector);
    let mut llama_position: llama_pos = 0;
    let mut final_outputs = Vec::new();

    for (segment_index, segment) in segments.iter().enumerate() {
        let is_final = segment_index.saturating_add(1) == segments.len();

        match segment {
            PromptSegment::Media(chunk) => {
                if is_final {
                    bail!("a decision prompt ends with text the answer is read from");
                }
                llama_position = chunk.eval(&mut context, llama_position, 0, n_batch, false)?;
            }
            PromptSegment::Text(text) => {
                let text_length = llama_pos::try_from(text.len())?;

                final_outputs = decode_decision_text(
                    &mut context,
                    prompt,
                    text.clone(),
                    llama_position,
                    is_final,
                )?;
                llama_position = llama_position.saturating_add(text_length);
            }
        }
    }

    Ok(prepared.read_answers(
        prompt,
        &context,
        DecisionPromptOutputs {
            batch_indices: &final_outputs,
            sequence_id: 0,
        },
    )?)
}

/// Answers `request` with the decision model `model` is.
///
/// Every prompt llama.cpp lays out is decoded in a context of its own, configured the way the
/// model's decision output requires, with the media of the request tokenized by `projector`.
///
/// # Errors
///
/// Fails when `model` serves no decisions, when llama.cpp refuses the request, or when a prompt
/// cannot be decoded or read.
pub fn answered_decisions(
    model: &Arc<LlamaModel>,
    backend: &LlamaBackend,
    context_params: LlamaContextParams,
    request: &DecisionRequest,
    projector: Option<&MtmdContext>,
) -> Result<IndexMap<String, DecisionAnswer>> {
    let decision_model = DecisionModel::new(Arc::clone(model))?;
    let mut prepared = PreparedDecisionRequest::new(&decision_model, request)?;
    let decision_context_params = decision_model.output().context_params(context_params);
    let mut answers = IndexMap::new();

    for prompt in prepared.prompts(projector)?.prompts {
        answers.extend(answers_of_prompt_decoded_alone(
            &mut prepared,
            model,
            backend,
            decision_context_params,
            &prompt,
            projector,
        )?);
    }

    Ok(answers)
}
