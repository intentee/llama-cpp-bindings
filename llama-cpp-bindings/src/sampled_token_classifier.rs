use std::collections::VecDeque;
use std::sync::Arc;

use llama_cpp_bindings_sys::llama_pos;
use llama_cpp_bindings_sys::llama_seq_id;

use llama_cpp_bindings_types::TokenUsage;
use llama_cpp_bindings_types::TokenUsageError;

use crate::bare_json_tool_calls::BareJsonToolCalls;
use crate::batch_add_error::BatchAddError;
use crate::context::LlamaContext;
use crate::error::EvalMultimodalChunksError;
use crate::error::SampleError;
use crate::error::TokenToStringError;
use crate::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use crate::generation_progress::GenerationProgress;
use crate::json_probe_outcome::JsonProbeOutcome;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::mtmd::MtmdContext;
use crate::mtmd::MtmdInputChunks;
use crate::mtmd::micro_batch_tokens;
use crate::sampled_token::SampledToken;
use crate::sampling::LlamaSampler;
use crate::streaming_json_probe::StreamingJsonProbe;
use crate::streaming_markers::StreamingMarkers;
use crate::token::LlamaToken;
use crate::token_piece::TokenPiece;

pub use crate::classified_sample::ClassifiedSample;
use crate::ingest_outcome::IngestOutcome;
pub use crate::sampled_token_section::SampledTokenSection;

#[derive(Copy, Clone, Debug)]
enum PendingMarkerStatus {
    Unmatched,
    ResolvedBoundary,
    AmbiguousBoundary,
}

#[derive(Clone, Debug)]
struct PendingToken {
    token: LlamaToken,
    decoded: String,
    section: SampledTokenSection,
    section_before_token: SampledTokenSection,
    marker_status: PendingMarkerStatus,
    is_from_prompt: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProbeMode {
    Idle,
    Active {
        probe: StreamingJsonProbe,
        held_count: usize,
    },
}

pub struct SampledTokenClassifier<'model> {
    model: &'model LlamaModel,
    markers: Arc<StreamingMarkers>,
    marker_lookback: usize,
    bare_json_tool_calls: BareJsonToolCalls,
    decoder: encoding_rs::Decoder,
    pending: VecDeque<PendingToken>,
    section: SampledTokenSection,
    pending_prompt_tokens: u64,
    usage: TokenUsage,
    probe_mode: ProbeMode,
}

impl<'model> SampledTokenClassifier<'model> {
    #[must_use]
    pub fn new(
        model: &'model LlamaModel,
        markers: Arc<StreamingMarkers>,
        bare_json_tool_calls: BareJsonToolCalls,
    ) -> Self {
        Self {
            model,
            marker_lookback: markers.max_token_len().saturating_sub(1),
            markers,
            bare_json_tool_calls,
            decoder: encoding_rs::UTF_8.new_decoder(),
            pending: VecDeque::new(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: TokenUsage::new(),
            probe_mode: ProbeMode::Idle,
        }
    }

    /// Classifies a generated token and appends every outcome it finalises to
    /// `outcomes`. An end-of-generation token is counted in the current section
    /// without being detokenised, and releases every token still held back.
    ///
    /// # Errors
    /// Returns [`TokenToStringError`] when the sampled token cannot be
    /// detokenised. The failure is surfaced rather than substituting an empty
    /// piece, so classification never silently drops generated text.
    pub fn ingest(
        &mut self,
        token: LlamaToken,
        outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<GenerationProgress, TokenToStringError> {
        if self.model.is_eog_token(&SampledToken::Content(token)) {
            self.record_usage_in(self.section);
            self.finish(outcomes);

            return Ok(GenerationProgress::Ended);
        }

        let decoded = self.decode(token)?;

        if self.markers.is_empty() {
            self.usage.record_undeterminable_token();
            outcomes.push(IngestOutcome {
                sampled_token: SampledToken::Undeterminable(token),
                piece: TokenPiece::Visible(decoded),
            });

            return Ok(GenerationProgress::Continues);
        }

        self.pending.push_back(PendingToken {
            token,
            decoded,
            section: self.section,
            section_before_token: self.section,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: false,
        });

        self.try_consume_marker_at_tail();
        self.classify_pending_tail(outcomes);
        self.drain_overflow(outcomes);

        Ok(GenerationProgress::Continues)
    }

    fn classify_pending_tail(&mut self, outcomes: &mut Vec<IngestOutcome>) {
        let Some(tail) = self.pending.back() else {
            return;
        };
        let section_disengages_probe = self.section_disengages_probe();
        let probe_engages =
            matches!(self.probe_mode, ProbeMode::Idle) && self.probe_engages_on(&tail.decoded);

        let probe_outcome = match &mut self.probe_mode {
            ProbeMode::Active { .. } if section_disengages_probe => {
                self.abandon_probe_held_before_tail(outcomes);

                return;
            }
            ProbeMode::Active { probe, held_count } => {
                *held_count += 1;

                probe.feed(&tail.decoded)
            }
            ProbeMode::Idle if probe_engages => {
                let mut probe = StreamingJsonProbe::default();
                let probe_outcome = probe.feed(&tail.decoded);

                self.probe_mode = ProbeMode::Active {
                    probe,
                    held_count: 1,
                };

                probe_outcome
            }
            ProbeMode::Idle => return,
        };

        match probe_outcome {
            JsonProbeOutcome::StillPossiblyValid => {}
            JsonProbeOutcome::CompletedValid => self.commit_probe_as_tool_call(outcomes),
            JsonProbeOutcome::Failed => self.abandon_probe(outcomes),
        }
    }

    const fn section_disengages_probe(&self) -> bool {
        matches!(
            self.section,
            SampledTokenSection::ToolCall | SampledTokenSection::Reasoning
        )
    }

    fn probe_engages_on(&self, piece: &str) -> bool {
        self.bare_json_tool_calls == BareJsonToolCalls::Detect
            && matches!(
                self.section,
                SampledTokenSection::Content | SampledTokenSection::Pending
            )
            && piece.trim_start().starts_with('{')
    }

    pub fn ingest_prompt_token(&mut self, token: LlamaToken) {
        if self.markers.is_empty() {
            return;
        }

        self.pending.push_back(PendingToken {
            token,
            decoded: String::new(),
            section: self.section,
            section_before_token: self.section,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: true,
        });

        self.try_consume_marker_at_tail();
        self.drain_overflow(&mut Vec::new());
    }

    pub fn ingest_prompt_tokens(&mut self, tokens: &[LlamaToken]) {
        if self.markers.is_empty() {
            return;
        }
        for &token in tokens {
            self.ingest_prompt_token(token);
        }
    }

    /// Releases every generated token still held back, for a generation that
    /// stops without an end-of-generation token (a token limit or a stop request).
    pub fn finish(&mut self, outcomes: &mut Vec<IngestOutcome>) {
        self.probe_mode = ProbeMode::Idle;

        while let Some(entry) = self.pending.pop_front() {
            if !entry.is_from_prompt {
                outcomes.push(self.finalize_entry(entry));
            }
        }
    }

    fn decode(&mut self, token: LlamaToken) -> Result<String, TokenToStringError> {
        self.model
            .token_to_piece(&SampledToken::Content(token), &mut self.decoder, true, None)
    }

    fn try_consume_marker_at_tail(&mut self) {
        let Some(marker) = self
            .markers
            .longest_matching_suffix(&self.pending.iter().rev().map(|entry| entry.token))
        else {
            return;
        };
        let span_start = self.pending.len() - marker.tokens().len();
        let span_section = marker.span_section(self.pending[span_start].section_before_token);
        let next_section = marker.next_section();
        let is_ambiguous_prefix = self.markers.is_prefix_of_longer_marker(marker.tokens());

        self.mark_marker_span(span_start, span_section, next_section, is_ambiguous_prefix);
    }

    fn mark_marker_span(
        &mut self,
        span_start: usize,
        span_section: SampledTokenSection,
        next_section: SampledTokenSection,
        is_ambiguous_prefix: bool,
    ) {
        let marker_status = if is_ambiguous_prefix {
            PendingMarkerStatus::AmbiguousBoundary
        } else {
            PendingMarkerStatus::ResolvedBoundary
        };
        for entry in self.pending.iter_mut().skip(span_start) {
            entry.marker_status = marker_status;
            entry.section = span_section;
        }

        self.section = next_section;
    }

    const fn probe_held_count(&self) -> usize {
        match self.probe_mode {
            ProbeMode::Active { held_count, .. } => held_count,
            ProbeMode::Idle => 0,
        }
    }

    fn drain_overflow(&mut self, outcomes: &mut Vec<IngestOutcome>) {
        let probe_held_count = self.probe_held_count();

        while let Some(front) = self.pending.front() {
            let drainable = self.pending.len().saturating_sub(probe_held_count);

            if drainable == 0 {
                break;
            }

            let beyond_lookback = drainable > self.marker_lookback;
            let resolved_boundary =
                matches!(front.marker_status, PendingMarkerStatus::ResolvedBoundary);

            if !resolved_boundary && !beyond_lookback {
                break;
            }

            let Some(entry) = self.pending.pop_front() else {
                break;
            };

            if !entry.is_from_prompt {
                outcomes.push(self.finalize_entry(entry));
            }
        }
    }

    fn take_probe_held_tokens(&mut self) -> VecDeque<PendingToken> {
        let probe_held_count = self.probe_held_count();

        self.probe_mode = ProbeMode::Idle;

        self.pending
            .split_off(self.pending.len().saturating_sub(probe_held_count))
    }

    fn commit_probe_as_tool_call(&mut self, outcomes: &mut Vec<IngestOutcome>) {
        self.section = SampledTokenSection::Content;

        for mut entry in self.take_probe_held_tokens() {
            entry.section = SampledTokenSection::ToolCall;
            outcomes.push(self.finalize_entry(entry));
        }
    }

    fn abandon_probe(&mut self, outcomes: &mut Vec<IngestOutcome>) {
        for entry in self.take_probe_held_tokens() {
            outcomes.push(self.finalize_entry(entry));
        }
    }

    fn abandon_probe_held_before_tail(&mut self, outcomes: &mut Vec<IngestOutcome>) {
        let Some(tail) = self.pending.pop_back() else {
            return;
        };

        self.abandon_probe(outcomes);
        self.pending.push_back(tail);
    }

    const fn record_usage_in(&mut self, section: SampledTokenSection) {
        match section {
            SampledTokenSection::Reasoning => self.usage.record_reasoning_token(),
            SampledTokenSection::Content => self.usage.record_content_token(),
            SampledTokenSection::ToolCall => self.usage.record_tool_call_token(),
            SampledTokenSection::Pending => self.usage.record_undeterminable_token(),
        }
    }

    fn finalize_entry(&mut self, entry: PendingToken) -> IngestOutcome {
        self.record_usage_in(entry.section);

        let sampled_token = match entry.section {
            SampledTokenSection::Reasoning => SampledToken::Reasoning(entry.token),
            SampledTokenSection::Content => SampledToken::Content(entry.token),
            SampledTokenSection::ToolCall => SampledToken::ToolCall(entry.token),
            SampledTokenSection::Pending => SampledToken::Undeterminable(entry.token),
        };

        let piece = match entry.marker_status {
            PendingMarkerStatus::Unmatched => TokenPiece::Visible(entry.decoded),
            PendingMarkerStatus::ResolvedBoundary | PendingMarkerStatus::AmbiguousBoundary => {
                TokenPiece::Marker(entry.decoded)
            }
        };

        IngestOutcome {
            sampled_token,
            piece,
        }
    }

    /// Samples a token and classifies it, appending the outcomes it finalises
    /// to `outcomes`.
    ///
    /// # Errors
    /// Forwards [`LlamaSampler::sample`] errors verbatim. Nothing is recorded on failure.
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
        self.ingest_prompt_token(token);
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
        self.ingest_prompt_tokens(tokens);
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
    /// Returns [`EvalMultimodalChunksError::EvalFailed`] when `params.n_batch` is not positive or
    /// a chunk cannot be decoded (both before any chunk is evaluated), or when evaluating a
    /// chunk fails,
    /// [`EvalMultimodalChunksError::UnknownChunkType`] when a chunk reports a
    /// type unknown to this binding, or
    /// [`EvalMultimodalChunksError::ChunkOutOfBounds`] when a valid index returns
    /// `None` from `chunks.get`.
    pub fn eval_multimodal_chunks(
        &mut self,
        chunks: &MtmdInputChunks,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        params: EvalMultimodalChunksParams,
    ) -> Result<llama_pos, EvalMultimodalChunksError> {
        chunks.fit_to_micro_batch(mtmd_ctx, micro_batch_tokens(llama_ctx, params.n_batch)?)?;

        let chunk_count = chunks.len();
        let mut next_position = params.start_position;

        for index in 0..chunk_count {
            let chunk = chunks
                .get(index)
                .ok_or(EvalMultimodalChunksError::ChunkOutOfBounds(index))?;
            let logits_for_this_chunk = params.logits_last && index + 1 == chunk_count;

            next_position = chunk.eval_single(
                mtmd_ctx,
                llama_ctx,
                next_position,
                params.seq_id,
                params.n_batch,
                logits_for_this_chunk,
            )?;
            crate::ingest_prompt_chunk::ingest_prompt_chunk(self, &chunk)?;
        }

        Ok(next_position)
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

    #[must_use]
    pub fn markers(&self) -> &StreamingMarkers {
        &self.markers
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::PendingMarkerStatus;
    use super::PendingToken;
    use super::ProbeMode;
    use super::SampledTokenClassifier;
    use crate::bare_json_tool_calls::BareJsonToolCalls;
    use crate::ingest_outcome::IngestOutcome;
    use crate::marker_role::MarkerRole;
    use crate::marker_role_candidate::MarkerRoleCandidate;
    use crate::sampled_token::SampledToken;
    use crate::sampled_token_section::SampledTokenSection;
    use crate::streaming_json_probe::StreamingJsonProbe;
    use crate::streaming_markers::StreamingMarkers;
    use crate::token::LlamaToken;

    fn token(id: i32) -> LlamaToken {
        LlamaToken::new(id)
    }

    fn markers_with(
        reasoning_open: Option<Vec<LlamaToken>>,
        reasoning_close: Option<Vec<LlamaToken>>,
    ) -> StreamingMarkers {
        let candidates = reasoning_open
            .into_iter()
            .map(|tokens| MarkerRoleCandidate {
                tokens,
                role: MarkerRole::ReasoningOpen,
            })
            .chain(
                reasoning_close
                    .into_iter()
                    .map(|tokens| MarkerRoleCandidate {
                        tokens,
                        role: MarkerRole::ReasoningClose,
                    }),
            );

        StreamingMarkers::from_candidates(candidates).expect("synthetic markers must be valid")
    }

    fn synthetic_classifier(markers: StreamingMarkers) -> SampledTokenClassifier<'static> {
        SampledTokenClassifier {
            model: unsafe { &*std::ptr::NonNull::<crate::model::LlamaModel>::dangling().as_ptr() },
            marker_lookback: markers.max_token_len().saturating_sub(1),
            bare_json_tool_calls: BareJsonToolCalls::Detect,
            markers: Arc::new(markers),
            decoder: encoding_rs::UTF_8.new_decoder(),
            pending: std::collections::VecDeque::new(),
            section: SampledTokenSection::Pending,
            pending_prompt_tokens: 0,
            usage: llama_cpp_bindings_types::TokenUsage::new(),
            probe_mode: ProbeMode::Idle,
        }
    }

    fn push_pending(classifier: &mut SampledTokenClassifier<'_>, token_id: i32, decoded: &str) {
        classifier.pending.push_back(PendingToken {
            token: token(token_id),
            decoded: decoded.to_owned(),
            section: classifier.section,
            section_before_token: classifier.section,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: false,
        });
    }

    fn push_pending_from_prompt(classifier: &mut SampledTokenClassifier<'_>, token_id: i32) {
        classifier.pending.push_back(PendingToken {
            token: token(token_id),
            decoded: String::new(),
            section: classifier.section,
            section_before_token: classifier.section,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: true,
        });
    }

    fn push_and_probe(
        classifier: &mut SampledTokenClassifier<'_>,
        token_id: i32,
        decoded: &str,
    ) -> Vec<IngestOutcome> {
        push_pending(classifier, token_id, decoded);
        classifier.try_consume_marker_at_tail();
        let mut outcomes = Vec::new();
        classifier.classify_pending_tail(&mut outcomes);
        classifier.drain_overflow(&mut outcomes);
        outcomes
    }

    fn drained(classifier: &mut SampledTokenClassifier<'_>) -> Vec<IngestOutcome> {
        let mut outcomes = Vec::new();
        classifier.drain_overflow(&mut outcomes);
        outcomes
    }

    fn finished(classifier: &mut SampledTokenClassifier<'_>) -> Vec<IngestOutcome> {
        let mut outcomes = Vec::new();
        classifier.finish(&mut outcomes);
        outcomes
    }

    fn outcome_pieces(outcomes: &[IngestOutcome]) -> Vec<&str> {
        outcomes
            .iter()
            .map(|outcome| outcome.piece.visible())
            .collect()
    }

    fn outcome_sections(outcomes: &[IngestOutcome]) -> Vec<SampledTokenSection> {
        outcomes
            .iter()
            .map(|outcome| match outcome.sampled_token {
                SampledToken::Reasoning(_) => SampledTokenSection::Reasoning,
                SampledToken::Content(_) => SampledTokenSection::Content,
                SampledToken::ToolCall(_) => SampledTokenSection::ToolCall,
                SampledToken::Undeterminable(_) => SampledTokenSection::Pending,
            })
            .collect()
    }

    fn markers_sharing_tool_call_open_with_a_reasoning_close(
        shared: Vec<LlamaToken>,
    ) -> StreamingMarkers {
        StreamingMarkers::from_candidates([
            MarkerRoleCandidate {
                tokens: vec![token(100)],
                role: MarkerRole::ReasoningOpen,
            },
            MarkerRoleCandidate {
                tokens: vec![token(200)],
                role: MarkerRole::ReasoningClose,
            },
            MarkerRoleCandidate {
                tokens: shared.clone(),
                role: MarkerRole::ReasoningClose,
            },
            MarkerRoleCandidate {
                tokens: shared,
                role: MarkerRole::ToolCallOpen,
            },
            MarkerRoleCandidate {
                tokens: vec![token(201)],
                role: MarkerRole::ToolCallClose,
            },
        ])
        .expect("synthetic markers must be valid")
    }

    #[test]
    fn a_token_that_is_both_a_reasoning_close_and_the_tool_call_open_opens_the_tool_call() {
        let mut classifier =
            synthetic_classifier(markers_sharing_tool_call_open_with_a_reasoning_close(vec![
                token(300),
            ]));
        classifier.section = SampledTokenSection::Content;

        push_pending(&mut classifier, 300, "<tool_call>");
        classifier.try_consume_marker_at_tail();

        assert_eq!(classifier.section, SampledTokenSection::ToolCall);
    }

    #[test]
    fn a_shared_marker_ends_reasoning_by_opening_the_tool_call() {
        let mut classifier =
            synthetic_classifier(markers_sharing_tool_call_open_with_a_reasoning_close(vec![
                token(300),
            ]));
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 300, "<tool_call>");
        classifier.try_consume_marker_at_tail();

        assert_eq!(classifier.section, SampledTokenSection::ToolCall);
    }

    #[test]
    fn longer_marker_reclassifies_a_completed_marker_that_was_its_prefix() {
        let markers = StreamingMarkers::from_candidates([
            MarkerRoleCandidate {
                tokens: vec![token(300)],
                role: MarkerRole::ReasoningClose,
            },
            MarkerRoleCandidate {
                tokens: vec![token(300), token(301)],
                role: MarkerRole::ToolCallOpen,
            },
        ])
        .expect("synthetic markers must be valid");
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 300, "<tool");
        classifier.try_consume_marker_at_tail();
        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert!(drained(&mut classifier).is_empty());

        push_pending(&mut classifier, 301, "_call>");
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(classifier.section, SampledTokenSection::ToolCall);
        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::ToolCall, SampledTokenSection::ToolCall]
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["", ""]);
    }

    #[test]
    fn longer_closing_marker_uses_section_from_before_its_shorter_prefix() {
        let markers = StreamingMarkers::from_candidates([
            MarkerRoleCandidate {
                tokens: vec![token(300)],
                role: MarkerRole::ReasoningClose,
            },
            MarkerRoleCandidate {
                tokens: vec![token(300), token(301)],
                role: MarkerRole::ReasoningClose,
            },
        ]);
        assert!(markers.is_ok());
        let markers = markers.unwrap_or_default();
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 300, "</think");
        classifier.try_consume_marker_at_tail();
        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert!(drained(&mut classifier).is_empty());

        push_pending(&mut classifier, 301, ">");
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Reasoning,
                SampledTokenSection::Reasoning
            ]
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["", ""]);
        assert_eq!(classifier.usage().reasoning_tokens, 2);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn single_token_close_marker_when_already_in_reasoning_emits_empty_piece_for_marker() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 7, "step");
        classifier.try_consume_marker_at_tail();
        let mut outcomes = drained(&mut classifier);

        push_pending(&mut classifier, 200, "</think>");
        classifier.try_consume_marker_at_tail();
        outcomes.extend(drained(&mut classifier));

        push_pending(&mut classifier, 9, "Hi");
        classifier.try_consume_marker_at_tail();
        outcomes.extend(drained(&mut classifier));

        outcomes.extend(finished(&mut classifier));

        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Reasoning,
                SampledTokenSection::Reasoning,
                SampledTokenSection::Content,
            ],
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["step", "", "Hi"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn multi_token_close_marker_suppresses_every_marker_token() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "r"), (200, "</"), (201, "thi"), (202, "nk>"), (9, "OK")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(drained(&mut classifier));
        }
        outcomes.extend(finished(&mut classifier));

        assert_eq!(outcome_pieces(&outcomes), vec!["r", "", "", "", "OK"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn marker_prefix_that_diverges_does_not_suppress_buffered_tokens() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "r"), (200, "a"), (201, "b"), (300, "x")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(drained(&mut classifier));
        }
        outcomes.extend(finished(&mut classifier));

        assert_eq!(outcome_pieces(&outcomes), vec!["r", "a", "b", "x"]);
        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::Reasoning(LlamaToken::new(0)))
        }));
        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
    }

    #[test]
    fn open_then_close_back_to_back_emits_two_empty_pieces_around_zero_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(100, "<think>"), (200, "</think>"), (9, "Hi")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(drained(&mut classifier));
        }
        outcomes.extend(finished(&mut classifier));

        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Reasoning,
                SampledTokenSection::Reasoning,
                SampledTokenSection::Content,
            ],
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["", "", "Hi"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn spurious_reasoning_close_in_content_section_classifies_as_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_pending(&mut classifier, 200, "</think>");
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Content],
        );
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn spurious_tool_call_close_in_reasoning_section_classifies_as_tool_call() {
        let markers = StreamingMarkers::from_candidates([
            MarkerRoleCandidate {
                tokens: vec![token(100)],
                role: MarkerRole::ReasoningOpen,
            },
            MarkerRoleCandidate {
                tokens: vec![token(200)],
                role: MarkerRole::ReasoningClose,
            },
            MarkerRoleCandidate {
                tokens: vec![token(300)],
                role: MarkerRole::ToolCallOpen,
            },
            MarkerRoleCandidate {
                tokens: vec![token(400)],
                role: MarkerRole::ToolCallClose,
            },
        ])
        .expect("synthetic markers must be valid");
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::ToolCall;

        push_pending(&mut classifier, 400, "</tool_call>");
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::ToolCall],
        );
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn finish_releases_every_held_generated_token() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_pending(&mut classifier, 7, "abc");
        push_pending(&mut classifier, 200, "</");
        push_pending(&mut classifier, 201, "th");

        let outcomes = finished(&mut classifier);

        assert_eq!(outcome_pieces(&outcomes), vec!["abc", "</", "th"]);
        assert!(classifier.pending.is_empty());
    }

    #[test]
    fn no_markers_marks_each_token_undeterminable_with_visible_piece() {
        let markers = StreamingMarkers::default();
        let mut classifier = synthetic_classifier(markers);

        push_pending(&mut classifier, 1, "h");
        push_pending(&mut classifier, 2, "i");
        let outcomes = finished(&mut classifier);

        assert_eq!(outcome_pieces(&outcomes), vec!["h", "i"]);
        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Pending, SampledTokenSection::Pending],
        );
    }

    #[test]
    fn ingest_prompt_tokens_without_markers_is_noop() {
        let markers = StreamingMarkers::default();
        let mut classifier = synthetic_classifier(markers);

        push_pending_from_prompt(&mut classifier, 7);
        push_pending_from_prompt(&mut classifier, 8);

        assert_eq!(classifier.section, SampledTokenSection::Pending);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_through_open_close_pair_ends_in_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7, 200] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            drained(&mut classifier);
        }

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_through_open_only_ends_in_reasoning() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            drained(&mut classifier);
        }

        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_does_not_record_usage() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7, 8, 9, 200, 201, 202, 11] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            drained(&mut classifier);
        }
        let drained = finished(&mut classifier);
        assert!(drained.is_empty());

        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn prompt_token_completing_marker_with_generated_token_is_suppressed_correctly() {
        let markers = markers_with(
            Some(vec![token(100)]),
            Some(vec![token(200), token(201), token(202)]),
        );
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        for token_id in [200, 201] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            drained(&mut classifier);
        }

        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
        assert_eq!(classifier.pending.len(), 2);

        classifier.pending.push_back(PendingToken {
            token: token(202),
            decoded: "k>".to_owned(),
            section: classifier.section,
            section_before_token: classifier.section,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: false,
        });
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(outcomes.len(), 1);
        assert_eq!(
            std::mem::discriminant(&outcomes[0].sampled_token),
            std::mem::discriminant(&SampledToken::Reasoning(LlamaToken::new(0)))
        );
        assert_eq!(outcomes[0].piece.visible(), "");
        assert_eq!(outcomes[0].piece.raw(), "k>");

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 1);
        assert_eq!(classifier.usage().content_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_with_multiple_round_trips_ends_in_content() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7, 200, 100, 8, 200] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            drained(&mut classifier);
        }

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);
        assert_eq!(classifier.usage().tool_call_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn ingest_prompt_tokens_initial_section_is_always_pending() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let classifier = synthetic_classifier(markers);

        assert_eq!(classifier.section, SampledTokenSection::Pending);
    }

    #[test]
    fn ingest_prompt_tokens_then_drain_for_generated_token_classifies_correctly() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);

        for token_id in [100, 7, 200] {
            push_pending_from_prompt(&mut classifier, token_id);
            classifier.try_consume_marker_at_tail();
            drained(&mut classifier);
        }

        assert_eq!(classifier.section, SampledTokenSection::Content);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().content_tokens, 0);

        classifier.pending.push_back(PendingToken {
            token: token(50),
            decoded: "hi".to_owned(),
            section: classifier.section,
            section_before_token: classifier.section,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: false,
        });
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(outcomes.len(), 1);
        assert_eq!(
            std::mem::discriminant(&outcomes[0].sampled_token),
            std::mem::discriminant(&SampledToken::Content(LlamaToken::new(0)))
        );
        assert_eq!(outcomes[0].piece.visible(), "hi");
        assert_eq!(classifier.usage().content_tokens, 1);
        assert_eq!(classifier.usage().reasoning_tokens, 0);
        assert_eq!(classifier.usage().undeterminable_tokens, 0);
    }

    #[test]
    fn close_marker_in_content_section_is_suppressed_as_boundary() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "hi"), (200, "</think>"), (8, "ok")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(drained(&mut classifier));
        }
        outcomes.extend(finished(&mut classifier));

        assert_eq!(
            outcome_sections(&outcomes),
            vec![
                SampledTokenSection::Content,
                SampledTokenSection::Content,
                SampledTokenSection::Content,
            ],
        );
        assert_eq!(outcome_pieces(&outcomes), vec!["hi", "", "ok"]);
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    #[test]
    fn open_marker_in_reasoning_section_is_suppressed_as_boundary() {
        let markers = markers_with(Some(vec![token(100)]), Some(vec![token(200)]));
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        let mut outcomes = Vec::new();
        for (id, decoded) in [(7, "step1"), (100, "<think>"), (8, "step2")] {
            push_pending(&mut classifier, id, decoded);
            classifier.try_consume_marker_at_tail();
            outcomes.extend(drained(&mut classifier));
        }
        outcomes.extend(finished(&mut classifier));

        assert_eq!(outcome_pieces(&outcomes), vec!["step1", "", "step2"]);
        assert_eq!(classifier.section, SampledTokenSection::Reasoning);
    }

    #[test]
    fn record_prompt_tokens_updates_usage() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);

        classifier.record_prompt_tokens(7);

        assert_eq!(classifier.usage().prompt_tokens, 7);
    }

    #[test]
    fn record_cached_prompt_tokens_updates_usage_when_under_limit() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);
        classifier.record_prompt_tokens(10);

        classifier.record_cached_prompt_tokens(3).unwrap();

        assert_eq!(classifier.usage().cached_prompt_tokens, 3);
    }

    #[test]
    fn record_cached_prompt_tokens_returns_error_when_over_prompt_total() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);
        classifier.record_prompt_tokens(2);

        let result = classifier.record_cached_prompt_tokens(5);

        assert!(result.is_err());
    }

    #[test]
    fn markers_accessor_returns_configured_markers() {
        let configured = markers_with(Some(vec![token(1)]), Some(vec![token(2)]));
        let classifier = synthetic_classifier(configured);

        let returned = classifier.markers();

        assert!(returned.iter().any(|marker| {
            marker.tokens() == [token(1)] && marker.roles() == [MarkerRole::ReasoningOpen]
        }));
        assert!(returned.iter().any(|marker| {
            marker.tokens() == [token(2)] && marker.roles() == [MarkerRole::ReasoningClose]
        }));
    }

    #[test]
    fn into_usage_consumes_classifier_and_yields_usage_snapshot() {
        let markers = markers_with(None, None);
        let mut classifier = synthetic_classifier(markers);
        classifier.record_prompt_tokens(11);

        let usage = classifier.into_usage();

        assert_eq!(usage.prompt_tokens, 11);
    }

    #[test]
    fn spurious_tool_call_close_in_content_section_classifies_as_content() {
        let markers = StreamingMarkers::from_candidates([MarkerRoleCandidate {
            tokens: vec![token(300)],
            role: MarkerRole::ToolCallClose,
        }])
        .expect("synthetic markers must be valid");
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_pending(&mut classifier, 300, "</tool_call>");
        classifier.try_consume_marker_at_tail();
        let outcomes = drained(&mut classifier);

        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Content],
        );
        assert_eq!(classifier.section, SampledTokenSection::Content);
    }

    fn markers_with_tool_call_open(tool_call_open: Vec<LlamaToken>) -> StreamingMarkers {
        StreamingMarkers::from_candidates([MarkerRoleCandidate {
            tokens: tool_call_open,
            role: MarkerRole::ToolCallOpen,
        }])
        .expect("synthetic markers must be valid")
    }

    fn feed_json_string(
        classifier: &mut SampledTokenClassifier<'_>,
        text: &str,
        starting_token_id: i32,
    ) -> Vec<IngestOutcome> {
        let mut outcomes = Vec::new();
        for (offset, ch) in text.char_indices() {
            let token_id = starting_token_id + i32::try_from(offset).unwrap_or(i32::MAX);
            let mut buffer = [0_u8; 4];
            let chunk = ch.encode_utf8(&mut buffer);
            outcomes.extend(push_and_probe(classifier, token_id, chunk));
        }
        outcomes
    }

    #[test]
    fn json_probe_engages_when_first_non_whitespace_is_open_brace_in_content() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_and_probe(&mut classifier, 1, "{");

        assert_ne!(classifier.probe_mode, ProbeMode::Idle);
    }

    #[test]
    fn json_probe_releases_tokens_as_tool_call_when_signature_matches() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, r#"{"name":"f","arguments":{}}"#, 100);

        assert!(!outcomes.is_empty());
        let sections = outcome_sections(&outcomes);
        assert!(
            sections
                .iter()
                .all(|section| *section == SampledTokenSection::ToolCall),
            "every emitted outcome should be ToolCall, got {sections:?}",
        );
        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
    }

    #[test]
    fn json_probe_releases_tokens_as_content_when_signature_does_not_match() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, r#"{"foo":"bar"}"#, 100);

        let sections = outcome_sections(&outcomes);
        assert!(
            sections
                .iter()
                .all(|section| *section == SampledTokenSection::Content),
            "every emitted outcome should be Content, got {sections:?}",
        );
        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
    }

    #[test]
    fn json_probe_releases_tokens_as_content_when_extra_top_level_key() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{},"extra":1}"#,
            100,
        );

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::Content(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_releases_tokens_as_content_when_arguments_is_not_object() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, r#"{"name":"f","arguments":"hi"}"#, 100);

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::Content(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_handles_strings_with_quoted_braces_in_arguments() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"q":"a } b"}}"#,
            100,
        );

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::ToolCall(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_handles_escaped_quotes_in_string_values() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"q":"he said \"hi\""}}"#,
            100,
        );

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::ToolCall(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_handles_unicode_letters_in_strings() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"日本語","arguments":{"city":"パリ"}}"#,
            100,
        );

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::ToolCall(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_handles_nested_objects() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"a":{"b":{"c":1}}}}"#,
            100,
        );

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::ToolCall(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_handles_arrays_inside_arguments() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            r#"{"name":"f","arguments":{"items":[1,2,3]}}"#,
            100,
        );

        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::ToolCall(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_does_not_engage_when_first_byte_is_close_brace() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(&mut classifier, "}}", 100);

        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
        assert!(outcomes.iter().all(|outcome| {
            std::mem::discriminant(&outcome.sampled_token)
                == std::mem::discriminant(&SampledToken::Content(LlamaToken::new(0)))
        }));
    }

    #[test]
    fn json_probe_does_not_engage_in_reasoning_section() {
        let markers = StreamingMarkers::from_candidates([
            MarkerRoleCandidate {
                tokens: vec![token(800)],
                role: MarkerRole::ReasoningOpen,
            },
            MarkerRoleCandidate {
                tokens: vec![token(801)],
                role: MarkerRole::ReasoningClose,
            },
            MarkerRoleCandidate {
                tokens: vec![token(900)],
                role: MarkerRole::ToolCallOpen,
            },
        ])
        .expect("synthetic markers must be valid");
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Reasoning;

        push_and_probe(&mut classifier, 1, "{");

        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
    }

    #[test]
    fn json_probe_does_not_engage_when_bare_json_tool_calls_are_ignored() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.bare_json_tool_calls = BareJsonToolCalls::Ignore;
        classifier.section = SampledTokenSection::Content;

        let outcomes = push_and_probe(&mut classifier, 1, "{");

        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
        assert_eq!(outcome_pieces(&outcomes), vec!["{"]);
    }

    #[test]
    fn json_probe_does_not_engage_in_tool_call_section() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::ToolCall;

        push_and_probe(&mut classifier, 1, "{");

        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
    }

    #[test]
    fn marker_probe_takes_precedence_when_both_could_match() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        outcomes.extend(push_and_probe(&mut classifier, 1, "{"));
        outcomes.extend(push_and_probe(&mut classifier, 900, r#"""#));

        assert_eq!(classifier.section, SampledTokenSection::ToolCall);
        assert_eq!(outcome_pieces(&outcomes), vec!["{", ""]);
        assert_eq!(
            outcome_sections(&outcomes),
            vec![SampledTokenSection::Content, SampledTokenSection::ToolCall],
        );
    }

    #[test]
    fn json_probe_consumes_two_consecutive_objects_separately() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let mut outcomes = Vec::new();
        outcomes.extend(feed_json_string(
            &mut classifier,
            r#"{"name":"a","arguments":{}}"#,
            100,
        ));
        outcomes.extend(feed_json_string(
            &mut classifier,
            r#"{"name":"b","arguments":{"x":1}}"#,
            200,
        ));

        let sections = outcome_sections(&outcomes);
        assert!(
            sections
                .iter()
                .all(|section| *section == SampledTokenSection::ToolCall),
            "two consecutive markerless tool calls must both classify as ToolCall, got {sections:?}",
        );
    }

    #[test]
    fn json_probe_with_leading_whitespace_then_open_brace_classifies_whitespace_as_content_and_json_as_tool_call()
     {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let outcomes = feed_json_string(
            &mut classifier,
            "\n  {\"name\":\"f\",\"arguments\":{}}",
            100,
        );

        let tool_call_count = outcomes
            .iter()
            .filter(|outcome| {
                std::mem::discriminant(&outcome.sampled_token)
                    == std::mem::discriminant(&SampledToken::ToolCall(LlamaToken::new(0)))
            })
            .count();
        let content_count = outcomes
            .iter()
            .filter(|outcome| {
                std::mem::discriminant(&outcome.sampled_token)
                    == std::mem::discriminant(&SampledToken::Content(LlamaToken::new(0)))
            })
            .count();
        assert_eq!(
            content_count, 3,
            "leading `\\n  ` should classify as content"
        );
        assert!(
            tool_call_count > 0,
            "the JSON object should classify as ToolCall",
        );
        assert_eq!(content_count + tool_call_count, outcomes.len());
    }

    #[test]
    fn json_probe_records_tool_call_token_usage_on_commit() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let json = r#"{"name":"f","arguments":{}}"#;
        let outcomes = feed_json_string(&mut classifier, json, 100);

        let emitted = outcomes.len();
        let usage = classifier.usage();
        assert_eq!(usage.tool_call_tokens, emitted as u64);
        assert_eq!(usage.content_tokens, 0);
    }

    #[test]
    fn json_probe_records_content_token_usage_on_abandon() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        let json = r#"{"foo":"bar"}"#;
        let outcomes = feed_json_string(&mut classifier, json, 100);

        let emitted = outcomes.len();
        let usage = classifier.usage();
        assert_eq!(usage.content_tokens, emitted as u64);
        assert_eq!(usage.tool_call_tokens, 0);
    }

    #[test]
    fn finish_during_active_json_probe_releases_held_tokens_as_content() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        push_and_probe(&mut classifier, 1, "{");
        push_and_probe(&mut classifier, 2, r#""name""#);
        assert_ne!(classifier.probe_mode, ProbeMode::Idle);

        let outcomes = finished(&mut classifier);

        let sections = outcome_sections(&outcomes);
        assert!(
            sections
                .iter()
                .all(|section| *section == SampledTokenSection::Content),
            "finishing mid-probe must release held tokens as Content, got {sections:?}",
        );
        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
    }

    #[test]
    fn commit_probe_as_tool_call_requeues_non_held_entries_and_releases_held_as_tool_call() {
        let markers = markers_with_tool_call_open(vec![token(900)]);
        let mut classifier = synthetic_classifier(markers);
        classifier.section = SampledTokenSection::Content;

        classifier.pending.push_back(PendingToken {
            token: token(1),
            decoded: "before".to_owned(),
            section: SampledTokenSection::Content,
            section_before_token: SampledTokenSection::Content,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: false,
        });
        classifier.pending.push_back(PendingToken {
            token: token(2),
            decoded: "{}".to_owned(),
            section: SampledTokenSection::Content,
            section_before_token: SampledTokenSection::Content,
            marker_status: PendingMarkerStatus::Unmatched,
            is_from_prompt: false,
        });
        classifier.probe_mode = ProbeMode::Active {
            probe: StreamingJsonProbe::default(),
            held_count: 1,
        };

        let mut outcomes = Vec::new();
        classifier.commit_probe_as_tool_call(&mut outcomes);

        let sections = outcome_sections(&outcomes);
        assert_eq!(sections, vec![SampledTokenSection::ToolCall]);
        assert_eq!(classifier.pending.len(), 1);
        assert_eq!(classifier.pending[0].token, token(1));
        assert_eq!(classifier.probe_mode, ProbeMode::Idle);
    }
}
