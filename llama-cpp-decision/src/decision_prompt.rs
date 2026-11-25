use std::ptr::NonNull;
use std::slice;

use llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_layout;
use llama_cpp_context::decision_order::DecisionOrder;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_input_chunk_view::MtmdInputChunkView;

use crate::decision_model::DecisionModel;
use crate::decision_prompt_handle::DecisionPromptHandle;

const fn layout_slice<'layout, TItem>(items: *const TItem, count: usize) -> &'layout [TItem] {
    if count == 0 {
        return &[];
    }

    unsafe { slice::from_raw_parts(items, count) }
}

/// One prompt llama.cpp laid out to answer a decision request.
pub struct DecisionPrompt {
    pub decision_model: DecisionModel,
    pub prompt: DecisionPromptHandle,
}

impl DecisionPrompt {
    /// The decision order llama.cpp lays the token at `position` out with.
    #[must_use]
    pub fn decision_order(&self, position: usize) -> DecisionOrder {
        let layout = self.layout();

        layout_slice(layout.orders, layout.orders_count)
            .get(position)
            .map_or(DecisionOrder::None, |order| DecisionOrder::from_raw(*order))
    }

    /// The first prompt position the answer is read at. Every position from it to the end of
    /// the prompt must be decoded in the prompt's final batch.
    #[must_use]
    pub fn first_read_position(&self) -> usize {
        self.layout().first_read_position
    }

    /// The media chunk that starts at `position` of the prompt, tokenized by `projector`, the
    /// projector the prompt was laid out with.
    #[must_use]
    pub fn media_chunk<'context>(
        &self,
        position: usize,
        projector: &'context MtmdContext,
    ) -> Option<MtmdInputChunkView<'_, 'context>> {
        NonNull::new(
            unsafe {
                llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_media_chunk(
                    self.prompt.handle.as_ptr(),
                    position,
                )
            }
            .cast_mut(),
        )
        .map(|chunk| unsafe { MtmdInputChunkView::from_raw(chunk, projector) })
    }

    /// Whether the whole prompt must be decoded in one batch.
    #[must_use]
    pub const fn must_be_decoded_at_once(&self) -> bool {
        !self.decision_model.has_causal_attention
    }

    /// Whether every token of the final batch is an output, and not only the last one.
    #[must_use]
    pub fn outputs_every_token(&self) -> bool {
        self.layout().outputs_every_token
    }

    #[must_use]
    pub fn tokens(&self) -> &[LlamaToken] {
        let layout = self.layout();

        layout_slice(layout.tokens.cast::<LlamaToken>(), layout.tokens_count)
    }

    fn layout(&self) -> llama_rs_decision_prompt_layout {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_layout_of(
                self.prompt.handle.as_ptr(),
            )
        }
    }
}
