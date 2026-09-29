use anyhow::Result;
use llama_cpp_bindings::GenerationProgress;
use llama_cpp_bindings::SampledTokenSection;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;

pub struct ClassifySampleLoop<'borrow, 'model, 'tokens> {
    pub classifier: &'borrow mut SampledTokenClassifier<'model>,
    pub sampler: &'borrow mut LlamaSampler,
    pub context: &'borrow mut LlamaContext<'model>,
    pub batch: &'borrow mut LlamaBatch<'tokens>,
    pub initial_position: i32,
    pub max_generated_tokens: i32,
}

#[derive(Debug, Default)]
pub struct ClassifySampleLoopOutcome {
    pub generated_raw: String,
    pub content_stream: String,
    pub reasoning_stream: String,
    pub observed_content: u64,
    pub observed_reasoning: u64,
    pub observed_tool_call: u64,
    pub observed_undeterminable: u64,
    pub eog_seen: bool,
}

impl ClassifySampleLoopOutcome {
    const fn record_end_of_generation(&mut self, section: SampledTokenSection) {
        self.eog_seen = true;

        match section {
            SampledTokenSection::Content => self.observed_content += 1,
            SampledTokenSection::Reasoning => self.observed_reasoning += 1,
            SampledTokenSection::ToolCall => self.observed_tool_call += 1,
            SampledTokenSection::Pending => self.observed_undeterminable += 1,
        }
    }

    fn record_outcome(&mut self, ingest: &IngestOutcome) {
        self.generated_raw.push_str(ingest.piece.raw());

        match ingest.sampled_token {
            SampledToken::Content(_) => {
                self.observed_content += 1;
                self.content_stream.push_str(ingest.piece.visible());
            }
            SampledToken::Reasoning(_) => {
                self.observed_reasoning += 1;
                self.reasoning_stream.push_str(ingest.piece.visible());
            }
            SampledToken::ToolCall(_) => self.observed_tool_call += 1,
            SampledToken::Undeterminable(_) => self.observed_undeterminable += 1,
        }
    }
}

impl ClassifySampleLoop<'_, '_, '_> {
    /// # Errors
    /// Forwards [`SampledTokenClassifier::sample`] / [`LlamaContext::decode`] /
    /// [`LlamaBatch::add`] errors verbatim. Stops on the end of generation, on
    /// `max_generated_tokens` exhaustion, or on the first error.
    pub fn run(self) -> Result<ClassifySampleLoopOutcome> {
        let mut outcome = ClassifySampleLoopOutcome::default();
        let mut ingest_outcomes = Vec::new();
        let mut position = self.initial_position;
        let max_position = position + self.max_generated_tokens;

        while position < max_position {
            let sampled = self.classifier.sample(
                self.sampler,
                self.context,
                self.batch.n_tokens() - 1,
                &mut ingest_outcomes,
            )?;

            if sampled.progress == GenerationProgress::Ended {
                outcome.record_end_of_generation(self.classifier.current_section());

                break;
            }

            self.batch.clear();
            self.batch
                .add(&SampledToken::Content(sampled.token), position, &[0], true)?;
            position += 1;

            self.context.decode(self.batch)?;
        }

        self.classifier.finish(&mut ingest_outcomes);

        for ingest_outcome in &ingest_outcomes {
            outcome.record_outcome(ingest_outcome);
        }

        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings::SampledTokenSection;
    use llama_cpp_bindings::TokenPiece;
    use llama_cpp_bindings::ingest_outcome::IngestOutcome;
    use llama_cpp_bindings::sampled_token::SampledToken;
    use llama_cpp_bindings::token::LlamaToken;

    use super::ClassifySampleLoopOutcome;

    #[test]
    fn records_a_tool_call_token_without_streaming_it() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome.record_outcome(&IngestOutcome {
            sampled_token: SampledToken::ToolCall(LlamaToken(42)),
            piece: TokenPiece::Visible("{".to_owned()),
        });

        assert_eq!(outcome.observed_tool_call, 1);
        assert_eq!(outcome.generated_raw, "{");
        assert!(outcome.content_stream.is_empty());
    }

    #[test]
    fn streams_the_visible_part_of_a_reasoning_token() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome.record_outcome(&IngestOutcome {
            sampled_token: SampledToken::Reasoning(LlamaToken(7)),
            piece: TokenPiece::Visible("thinking".to_owned()),
        });

        assert_eq!(outcome.observed_reasoning, 1);
        assert_eq!(outcome.reasoning_stream, "thinking");
    }

    #[test]
    fn counts_an_undeterminable_token_without_streaming_it() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome.record_outcome(&IngestOutcome {
            sampled_token: SampledToken::Undeterminable(LlamaToken(9)),
            piece: TokenPiece::Visible("ignored".to_owned()),
        });

        assert_eq!(outcome.observed_undeterminable, 1);
        assert!(outcome.content_stream.is_empty());
        assert!(outcome.reasoning_stream.is_empty());
    }

    #[test]
    fn counts_the_end_of_generation_in_the_section_it_ended() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome.record_end_of_generation(SampledTokenSection::Content);

        assert!(outcome.eog_seen);
        assert_eq!(outcome.observed_content, 1);
    }
}
