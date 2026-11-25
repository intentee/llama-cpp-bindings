use std::collections::VecDeque;

use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_bindings_sys::bindings::llama_seq_id;
use llama_cpp_chat::chat_output::ChatOutput;
use llama_cpp_chat::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_chat::chat_piece_role::ChatPieceRole;
use llama_cpp_chat::chat_reading::ChatReading;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_model::sampled_token_section::SampledTokenSection;
use llama_cpp_mtmd::evaluated_mtmd_input_chunk::EvaluatedMtmdInputChunk;
use llama_cpp_mtmd::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use llama_cpp_mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_token_usage::token_usage::TokenUsage;

use crate::chat_output_reading::ChatOutputReading;
use crate::chat_session_attribution::ChatSessionAttribution;
use crate::classified_sample::ClassifiedSample;
use crate::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use crate::generation_progress::GenerationProgress;
use crate::ingest_outcome::IngestOutcome;
use crate::llama_cpp_error::LlamaCppError;
use crate::pending_chat_piece::PendingChatPiece;
use crate::token_attribution::TokenAttribution;
use crate::token_piece::TokenPiece;

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
pub struct SampledTokenClassifier<'model> {
    model: &'model LlamaModel,
    attribution: TokenAttribution,
    decoder: encoding_rs::Decoder,
    section: SampledTokenSection,
    pending_prompt_tokens: u64,
    usage: TokenUsage,
}

impl<'model> SampledTokenClassifier<'model> {
    /// A classifier for the answer to `session`, which was created for `model`. The classifier
    /// owns the session for as long as the answer is generated, starting from the session's
    /// reading of an answer with nothing generated yet.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::EmptyAnswerReading`] when llama.cpp fails while reading the empty
    /// answer.
    pub fn for_chat_session(
        model: &'model LlamaModel,
        session: ChatSession,
    ) -> Result<Self, LlamaCppError> {
        let output = ChatOutput::default();
        let latest_reading =
            ChatOutputReading::read(&session, &output, ChatOutputCompleteness::Partial)
                .map_err(LlamaCppError::EmptyAnswerReading)?;

        Ok(Self {
            model,
            attribution: TokenAttribution::ChatSession(Box::new(ChatSessionAttribution {
                preserved_tokens: session.preserved_tokens(),
                session,
                output,
                pending_pieces: VecDeque::new(),
                latest_reading,
            })),
            decoder: encoding_rs::UTF_8.new_decoder(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
        })
    }

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

    pub const fn commit_prompt_tokens(&mut self) -> u64 {
        let promoted = self.pending_prompt_tokens;
        self.usage.record_prompt_tokens(promoted);
        self.pending_prompt_tokens = 0;

        promoted
    }

    #[must_use]
    pub const fn current_section(&self) -> SampledTokenSection {
        self.section
    }

    pub const fn discard_pending_prompt_tokens(&mut self) -> u64 {
        let discarded = self.pending_prompt_tokens;
        self.pending_prompt_tokens = 0;

        discarded
    }

    /// Checks every chunk and its positions before evaluating any of them, as
    /// [`MtmdInputChunks::eval_chunks`] does, so a chunk that cannot be decoded moves neither the
    /// KV cache nor the usage counters, then ingests the usage of each chunk right after
    /// evaluating it.
    ///
    /// # Errors
    /// Returns [`LlamaCppError::MultimodalChunksEval`] when `llama_ctx` was created for another
    /// model than the projector of `chunks`, `params.n_batch` or `params.seq_id` does not fit the
    /// context, a chunk does not fit the decodes of `params.n_batch` in the context or reports a
    /// type unknown to this binding, `params.start_position` is negative, or a chunk has positions
    /// that overflow from `params.start_position` (all before any chunk is evaluated), or when
    /// evaluating a chunk fails, after which no further chunk is evaluated.
    pub fn eval_multimodal_chunks(
        &mut self,
        chunks: &MtmdInputChunks<'_>,
        llama_ctx: &mut LlamaContext,
        EvalMultimodalChunksParams {
            start_position,
            seq_id,
            n_batch,
            logits_last,
        }: EvalMultimodalChunksParams,
    ) -> Result<llama_pos, LlamaCppError> {
        let fitted_chunks = chunks
            .fit_in(llama_ctx, n_batch, seq_id)
            .map_err(LlamaCppError::MultimodalChunksEval)?;

        fitted_chunks
            .eval_each(llama_ctx, start_position, seq_id, n_batch, logits_last)
            .and_then(|mut evaluated_chunks| {
                evaluated_chunks.try_fold(
                    start_position,
                    |_position_before_chunk, evaluated_chunk| {
                        let EvaluatedMtmdInputChunk {
                            fitted_chunk,
                            next_position,
                        } = evaluated_chunk?;

                        self.ingest_fitted_chunk(fitted_chunk);

                        Ok(next_position)
                    },
                )
            })
            .map_err(LlamaCppError::MultimodalChunksEval)
    }

    /// # Errors
    /// Forwards [`LlamaBatch::add_sequence`] errors as [`LlamaCppError::Context`]. Nothing is staged on failure.
    pub fn feed_prompt_sequence_to_batch(
        &mut self,
        batch: &mut LlamaBatch,
        tokens: &[LlamaToken],
        seq_id: llama_seq_id,
        logits_all: bool,
    ) -> Result<(), LlamaCppError> {
        batch.add_sequence(tokens, seq_id, logits_all)?;
        self.pending_prompt_tokens = self
            .pending_prompt_tokens
            .saturating_add(tokens.len() as u64);

        Ok(())
    }

    /// # Errors
    /// Forwards [`LlamaBatch::add`] errors as [`LlamaCppError::Context`]. Nothing is staged on failure.
    pub fn feed_prompt_to_batch(
        &mut self,
        batch: &mut LlamaBatch,
        token: LlamaToken,
        position: llama_pos,
        seq_ids: &[llama_seq_id],
        logits: bool,
    ) -> Result<(), LlamaCppError> {
        batch.add(SampledToken::Content(token), position, seq_ids, logits)?;
        self.pending_prompt_tokens = self.pending_prompt_tokens.saturating_add(1);

        Ok(())
    }

    /// Settles every generated token still pending, for a generation that stops without an
    /// end-of-generation token (a token limit or a stop request).
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::Classify`] when llama.cpp fails while parsing the finished answer.
    pub fn finish(&mut self, outcomes: &mut Vec<IngestOutcome>) -> Result<(), LlamaCppError> {
        self.attribute(ChatOutputCompleteness::Finished, outcomes)
    }

    /// Classifies a generated token and appends every outcome it settles to `outcomes`. An
    /// end-of-generation token settles every token still pending and is counted in the section
    /// generation ended in.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaCppError::Detokenize`] when the token cannot be detokenized, or
    /// [`LlamaCppError::Classify`] when llama.cpp fails while parsing the answer.
    pub fn ingest(
        &mut self,
        token: LlamaToken,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<GenerationProgress, LlamaCppError> {
        if self.model.is_eog_token(&SampledToken::Content(token)) {
            self.finish(outcomes)?;
            self.record_usage_in(self.section);

            return Ok(GenerationProgress::Ended);
        }

        let attribution = match &mut self.attribution {
            TokenAttribution::Undeterminable => {
                let piece = self
                    .model
                    .token_to_piece(SampledToken::Content(token), &mut self.decoder, true, None)
                    .map_err(LlamaCppError::Detokenize)?;
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
        let piece = self
            .model
            .token_to_piece(
                SampledToken::Content(token),
                &mut self.decoder,
                renders_special,
                None,
            )
            .map_err(LlamaCppError::Detokenize)?;

        let bytes = attribution.output.push(token, &piece);

        attribution.pending_pieces.push_back(PendingChatPiece {
            token,
            bytes,
            piece,
        });
        self.attribute(ChatOutputCompleteness::Partial, outcomes)?;

        Ok(GenerationProgress::Continues)
    }

    pub fn ingest_fitted_chunk(&mut self, fitted_chunk: &FittedMtmdInputChunk<'_, '_>) {
        let n_tokens = fitted_chunk.chunk().n_tokens() as u64;

        match fitted_chunk.chunk_type() {
            MtmdInputChunkType::Text => self.record_prompt_tokens(n_tokens),
            MtmdInputChunkType::Image => self.record_input_image_tokens(n_tokens),
            MtmdInputChunkType::Audio => self.record_input_audio_tokens(n_tokens),
        }
    }

    #[must_use]
    pub fn into_usage(self) -> TokenUsage {
        self.usage
    }

    /// The template parser's latest reading of the answer, the one that attributed the most
    /// recent outcomes, or `None` for generation outside of a chat template. Parsing it maps
    /// that same parse into the answer's message.
    #[must_use]
    pub fn latest_reading(&self) -> Option<&ChatReading> {
        match &self.attribution {
            TokenAttribution::Undeterminable => None,
            TokenAttribution::ChatSession(attribution) => Some(&attribution.latest_reading.reading),
        }
    }

    #[must_use]
    pub const fn pending_prompt_tokens(&self) -> u64 {
        self.pending_prompt_tokens
    }

    /// # Errors
    /// Returns [`LlamaCppError::TokenUsage`] when the running cached total would exceed the
    /// prompt total.
    pub fn record_cached_prompt_tokens(&mut self, count: u64) -> Result<(), LlamaCppError> {
        Ok(self.usage.record_cached_prompt_tokens(count)?)
    }

    pub const fn record_input_audio_tokens(&mut self, count: u64) {
        self.usage.record_input_audio_tokens(count);
    }

    pub const fn record_input_image_tokens(&mut self, count: u64) {
        self.usage.record_input_image_tokens(count);
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.usage.record_prompt_tokens(count);
    }

    /// Samples a token and classifies it, appending the outcomes it settles to `outcomes`.
    ///
    /// # Errors
    /// Forwards [`LlamaSampler::sample`] errors as [`LlamaCppError::Sampler`] and wraps
    /// [`Self::ingest`] errors in [`LlamaCppError::SampleIngest`].
    pub fn sample(
        &mut self,
        sampler: &mut LlamaSampler,
        context: &LlamaContext,
        idx: i32,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<ClassifiedSample, LlamaCppError> {
        let token = sampler.sample(context, idx)?;
        let progress = self
            .ingest(token, outcomes)
            .map_err(Box::new)
            .map_err(LlamaCppError::SampleIngest)?;

        Ok(ClassifiedSample { token, progress })
    }

    #[must_use]
    pub const fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    fn attribute(
        &mut self,
        completeness: ChatOutputCompleteness,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<(), LlamaCppError> {
        let TokenAttribution::ChatSession(attribution) = &mut self.attribution else {
            return Ok(());
        };
        let attribution = &mut **attribution;

        if !attribution
            .latest_reading
            .has_read(&attribution.output, completeness)
        {
            attribution.latest_reading =
                ChatOutputReading::read(&attribution.session, &attribution.output, completeness)
                    .map_err(LlamaCppError::Classify)?;
        }

        let classification = attribution.latest_reading.reading.classification();
        let mut released = Vec::new();

        while let Some(pending_piece) = attribution.pending_pieces.pop_front() {
            let Some(role) = classification.classify_token(&pending_piece.bytes) else {
                attribution.pending_pieces.push_front(pending_piece);
                break;
            };

            released.push(outcome_of(pending_piece.token, role, pending_piece.piece));
        }

        for outcome in released {
            self.section = outcome.sampled_token.section();
            self.record_usage_in(self.section);
            outcomes.push(outcome);
        }

        Ok(())
    }

    const fn record_usage_in(&mut self, section: SampledTokenSection) {
        match section {
            SampledTokenSection::Reasoning => self.usage.record_reasoning_token(),
            SampledTokenSection::Content => self.usage.record_content_token(),
            SampledTokenSection::ToolCall => self.usage.record_tool_call_token(),
            SampledTokenSection::Pending => self.usage.record_undeterminable_token(),
        }
    }
}
