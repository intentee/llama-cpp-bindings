use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_sys::llama_seq_id;
use llama_cpp_bindings_types::TokenUsage;
use llama_cpp_bindings_types::TokenUsageError;

use crate::bare_json_tool_call_probe::BareJsonToolCallProbe;
use crate::bare_json_tool_calls::BareJsonToolCalls;
use crate::batch_add_error::BatchAddError;
use crate::chat_output::ChatOutput;
use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_piece_role::ChatPieceRole;
use crate::chat_session::ChatSession;
use crate::classified_sample::ClassifiedSample;
use crate::context::LlamaContext;
use crate::error::eval_multimodal_chunks_error::EvalMultimodalChunksError;
use crate::error::ingest_error::IngestError;
use crate::error::sample_error::SampleError;
use crate::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use crate::generation_progress::GenerationProgress;
use crate::ingest_outcome::IngestOutcome;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::mtmd::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use crate::mtmd::fitted_mtmd_input_chunks::FittedMtmdInputChunks;
use crate::mtmd::mtmd_context::MtmdContext;
use crate::mtmd::mtmd_eval_limits::MtmdEvalLimits;
use crate::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::mtmd::mtmd_input_chunks::MtmdInputChunks;
use crate::sampled_token::SampledToken;
use crate::sampled_token_section::SampledTokenSection;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::token_piece::TokenPiece;

enum ContentProbe {
    Disabled,
    BareJson(BareJsonToolCallProbe),
}

impl ContentProbe {
    fn pass(&mut self, outcome: IngestOutcome, released: &mut Vec<IngestOutcome>) {
        match self {
            Self::Disabled => released.push(outcome),
            Self::BareJson(probe) => probe.pass(outcome, released),
        }
    }

    fn release(&mut self, released: &mut Vec<IngestOutcome>) {
        match self {
            Self::Disabled => {}
            Self::BareJson(probe) => probe.release(released),
        }
    }
}

struct ChatSessionAttribution<'session, 'model> {
    session: &'session ChatSession<'model>,
    preserved_tokens: Vec<LlamaToken>,
    output: ChatOutput,
    attributed_tokens: usize,
    content_probe: ContentProbe,
}

enum TokenAttribution<'session, 'model> {
    Undeterminable,
    ChatSession(ChatSessionAttribution<'session, 'model>),
}

const fn outcome_of(token: LlamaToken, role: ChatPieceRole, piece: String) -> IngestOutcome {
    IngestOutcome {
        sampled_token: SampledToken::in_section(role.section(), token),
        piece: match role {
            ChatPieceRole::Text(_) => TokenPiece::Visible(piece),
            ChatPieceRole::Marker(_) => TokenPiece::Marker(piece),
        },
    }
}

/// Attributes every generated token to reasoning, content, or a tool call, the way llama.cpp's
/// parse of the chat template attributes the text the token decoded to, and counts the tokens
/// of each section.
pub struct SampledTokenClassifier<'session, 'model> {
    model: &'model LlamaModel,
    attribution: TokenAttribution<'session, 'model>,
    decoder: encoding_rs::Decoder,
    section: SampledTokenSection,
    pending_prompt_tokens: u64,
    usage: TokenUsage,
}

impl<'session, 'model> SampledTokenClassifier<'session, 'model> {
    /// A classifier for generation outside of a chat template, whose tokens belong to no
    /// section.
    #[must_use]
    pub fn undeterminable(model: &'model LlamaModel) -> Self {
        Self {
            model,
            attribution: TokenAttribution::Undeterminable,
            decoder: encoding_rs::UTF_8.new_decoder(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
        }
    }

    /// A classifier for the answer to `session`. When the session's template cannot render tool
    /// calls and `bare_json_tool_calls` asks for it, content that is a bare JSON tool call object
    /// is attributed to a tool call.
    #[must_use]
    pub fn for_chat_session(
        session: &'session ChatSession<'model>,
        bare_json_tool_calls: BareJsonToolCalls,
    ) -> Self {
        let content_probe = if !session.supports_tool_calls()
            && bare_json_tool_calls == BareJsonToolCalls::Detect
        {
            ContentProbe::BareJson(BareJsonToolCallProbe::default())
        } else {
            ContentProbe::Disabled
        };

        Self {
            model: session.model(),
            attribution: TokenAttribution::ChatSession(ChatSessionAttribution {
                session,
                preserved_tokens: session.preserved_tokens(),
                output: ChatOutput::default(),
                attributed_tokens: 0,
                content_probe,
            }),
            decoder: encoding_rs::UTF_8.new_decoder(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
        }
    }

    /// Classifies a generated token and appends every outcome it settles to `outcomes`. An
    /// end-of-generation token settles every token still pending and is counted in the section
    /// generation ended in.
    ///
    /// # Errors
    ///
    /// Returns [`IngestError`] when the token cannot be detokenized or llama.cpp fails while
    /// parsing the answer.
    pub fn ingest(
        &mut self,
        token: LlamaToken,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<GenerationProgress, IngestError> {
        if self.model.is_eog_token(&SampledToken::Content(token)) {
            self.finish(outcomes)?;
            self.record_usage_in(self.section);

            return Ok(GenerationProgress::Ended);
        }

        let attribution = match &mut self.attribution {
            TokenAttribution::Undeterminable => {
                let piece = self.model.token_to_piece(
                    &SampledToken::Content(token),
                    &mut self.decoder,
                    true,
                    None,
                )?;
                self.usage.record_undeterminable_token();
                outcomes.push(IngestOutcome {
                    sampled_token: SampledToken::Undeterminable(token),
                    piece: TokenPiece::Visible(piece),
                });

                return Ok(GenerationProgress::Continues);
            }
            TokenAttribution::ChatSession(attribution) => attribution,
        };
        let renders_special = attribution.preserved_tokens.contains(&token);
        let piece = self.model.token_to_piece(
            &SampledToken::Content(token),
            &mut self.decoder,
            renders_special,
            None,
        )?;

        attribution.output.push(token, &piece);
        self.attribute(ChatOutputCompleteness::Partial, outcomes)?;

        Ok(GenerationProgress::Continues)
    }

    fn attribute(
        &mut self,
        completeness: ChatOutputCompleteness,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<(), IngestError> {
        let TokenAttribution::ChatSession(attribution) = &mut self.attribution else {
            return Ok(());
        };
        let classification = attribution
            .session
            .classify(&attribution.output, completeness)?;
        let mut released = Vec::new();

        while let Some(output_token) = attribution
            .output
            .tokens()
            .get(attribution.attributed_tokens)
        {
            let Some(role) = classification.classify_token(&output_token.bytes) else {
                break;
            };
            let piece = attribution.output.text()[output_token.bytes.clone()].to_owned();

            attribution
                .content_probe
                .pass(outcome_of(output_token.token, role, piece), &mut released);
            attribution.attributed_tokens += 1;
        }

        if completeness == ChatOutputCompleteness::Finished {
            attribution.content_probe.release(&mut released);
        }

        for outcome in released {
            self.section = outcome.sampled_token.section();
            self.record_usage_in(self.section);
            outcomes.push(outcome);
        }

        Ok(())
    }

    /// Settles every generated token still pending, for a generation that stops without an
    /// end-of-generation token (a token limit or a stop request).
    ///
    /// # Errors
    ///
    /// Returns [`IngestError`] when llama.cpp fails while parsing the finished answer.
    pub fn finish(&mut self, outcomes: &mut Vec<IngestOutcome>) -> Result<(), IngestError> {
        self.attribute(ChatOutputCompleteness::Finished, outcomes)
    }

    const fn record_usage_in(&mut self, section: SampledTokenSection) {
        match section {
            SampledTokenSection::Reasoning => self.usage.record_reasoning_token(),
            SampledTokenSection::Content => self.usage.record_content_token(),
            SampledTokenSection::ToolCall => self.usage.record_tool_call_token(),
            SampledTokenSection::Pending => self.usage.record_undeterminable_token(),
        }
    }

    /// Samples a token and classifies it, appending the outcomes it settles to `outcomes`.
    ///
    /// # Errors
    /// Forwards [`LlamaSampler::sample`] and [`Self::ingest`] errors verbatim.
    pub fn sample(
        &mut self,
        sampler: &mut LlamaSampler,
        context: &LlamaContext,
        idx: i32,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<ClassifiedSample, SampleError> {
        let token = sampler.sample(context, idx)?;
        let progress = self.ingest(token, outcomes)?;

        Ok(ClassifiedSample { token, progress })
    }

    /// # Errors
    /// Forwards [`LlamaBatch::add`] errors verbatim. Nothing is staged on failure.
    pub fn feed_prompt_to_batch(
        &mut self,
        batch: &mut LlamaBatch,
        token: LlamaToken,
        position: llama_pos,
        seq_ids: &[llama_seq_id],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        batch.add(&SampledToken::Content(token), position, seq_ids, logits)?;
        self.pending_prompt_tokens = self.pending_prompt_tokens.saturating_add(1);

        Ok(())
    }

    /// # Errors
    /// Forwards [`LlamaBatch::add_sequence`] errors verbatim. Nothing is staged on failure.
    pub fn feed_prompt_sequence_to_batch(
        &mut self,
        batch: &mut LlamaBatch,
        tokens: &[LlamaToken],
        seq_id: llama_seq_id,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        batch.add_sequence(tokens, seq_id, logits_all)?;
        self.pending_prompt_tokens = self
            .pending_prompt_tokens
            .saturating_add(tokens.len() as u64);

        Ok(())
    }

    pub const fn commit_prompt_tokens(&mut self) -> u64 {
        let promoted = self.pending_prompt_tokens;
        self.usage.record_prompt_tokens(promoted);
        self.pending_prompt_tokens = 0;

        promoted
    }

    pub const fn discard_pending_prompt_tokens(&mut self) -> u64 {
        let discarded = self.pending_prompt_tokens;
        self.pending_prompt_tokens = 0;

        discarded
    }

    #[must_use]
    pub const fn pending_prompt_tokens(&self) -> u64 {
        self.pending_prompt_tokens
    }

    /// Checks every chunk with [`MtmdInputChunks::fit_to_micro_batch`] before evaluating any of
    /// them, so a chunk that cannot be decoded moves neither the KV cache nor the usage counters.
    ///
    /// # Errors
    /// Returns [`EvalMultimodalChunksError::EvalFailed`] when `params.n_batch` or `params.seq_id`
    /// does not fit the context, a chunk cannot be decoded or reports a type unknown to this
    /// binding (all before any chunk is evaluated), or when evaluating a chunk fails, or
    /// [`EvalMultimodalChunksError::ChunkOutOfBounds`] when a valid index returns
    /// `None` from `chunks.get`.
    pub fn eval_multimodal_chunks(
        &mut self,
        chunks: &MtmdInputChunks,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        params: EvalMultimodalChunksParams,
    ) -> Result<llama_pos, EvalMultimodalChunksError> {
        let limits = MtmdEvalLimits::of(llama_ctx);
        let fitted_chunks = FittedMtmdInputChunks::fit(
            chunks,
            mtmd_ctx,
            limits
                .context
                .micro_batch_tokens(limits.batch_tokens(params.n_batch, params.seq_id)?),
        )?;
        let chunk_count = fitted_chunks.len();
        let mut next_position = params.start_position;

        for index in 0..chunk_count {
            let fitted_chunk = FittedMtmdInputChunk::at(&fitted_chunks, index)
                .ok_or(EvalMultimodalChunksError::ChunkOutOfBounds(index))?;
            let logits_for_this_chunk = params.logits_last && index + 1 == chunk_count;

            next_position = fitted_chunk.eval_single(
                mtmd_ctx,
                llama_ctx,
                next_position,
                params.seq_id,
                params.n_batch,
                logits_for_this_chunk,
            )?;
            self.ingest_fitted_chunk(&fitted_chunk);
        }

        Ok(next_position)
    }

    pub fn ingest_fitted_chunk(&mut self, fitted_chunk: &FittedMtmdInputChunk<'_>) {
        let n_tokens = fitted_chunk.chunk().n_tokens() as u64;

        match fitted_chunk.chunk_type() {
            MtmdInputChunkType::Text => self.record_prompt_tokens(n_tokens),
            MtmdInputChunkType::Image => self.record_input_image_tokens(n_tokens),
            MtmdInputChunkType::Audio => self.record_input_audio_tokens(n_tokens),
        }
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.usage.record_prompt_tokens(count);
    }

    pub const fn record_input_image_tokens(&mut self, count: u64) {
        self.usage.record_input_image_tokens(count);
    }

    pub const fn record_input_audio_tokens(&mut self, count: u64) {
        self.usage.record_input_audio_tokens(count);
    }

    /// # Errors
    /// Forwards [`TokenUsageError::CachedExceedsPrompt`] when the running cached total would
    /// exceed the prompt total.
    pub const fn record_cached_prompt_tokens(&mut self, count: u64) -> Result<(), TokenUsageError> {
        self.usage.record_cached_prompt_tokens(count)
    }

    #[must_use]
    pub const fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    #[must_use]
    pub fn into_usage(self) -> TokenUsage {
        self.usage
    }

    #[must_use]
    pub const fn current_section(&self) -> SampledTokenSection {
        self.section
    }
}
