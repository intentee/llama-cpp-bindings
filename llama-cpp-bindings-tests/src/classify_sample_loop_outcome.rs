use llama_cpp_bindings::ChatOutput;
use llama_cpp_bindings::SampledTokenSection;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token::SampledToken;

#[derive(Debug, Default)]
pub struct ClassifySampleLoopOutcome {
    pub generated: ChatOutput,
    pub content_stream: String,
    pub reasoning_stream: String,
    pub observed_content: u64,
    pub observed_reasoning: u64,
    pub observed_tool_call: u64,
    pub observed_undeterminable: u64,
    pub eog_seen: bool,
}

impl ClassifySampleLoopOutcome {
    pub const fn record_end_of_generation(&mut self, section: SampledTokenSection) {
        self.eog_seen = true;

        match section {
            SampledTokenSection::Content => self.observed_content += 1,
            SampledTokenSection::Reasoning => self.observed_reasoning += 1,
            SampledTokenSection::ToolCall => self.observed_tool_call += 1,
            SampledTokenSection::Pending => self.observed_undeterminable += 1,
        }
    }

    pub fn record_outcome(&mut self, ingest: &IngestOutcome) {
        self.generated
            .push(ingest.sampled_token.token(), ingest.piece.raw());

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
        assert_eq!(outcome.generated.text(), "{");
        assert_eq!(outcome.content_stream, "");
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
        assert_eq!(outcome.content_stream, "");
        assert_eq!(outcome.reasoning_stream, "");
    }

    #[test]
    fn counts_the_end_of_generation_in_the_section_it_ended() {
        let counts = [
            SampledTokenSection::Content,
            SampledTokenSection::Reasoning,
            SampledTokenSection::ToolCall,
            SampledTokenSection::Pending,
        ]
        .map(|section| {
            let mut outcome = ClassifySampleLoopOutcome::default();

            outcome.record_end_of_generation(section);

            [
                u64::from(outcome.eog_seen),
                outcome.observed_content,
                outcome.observed_reasoning,
                outcome.observed_tool_call,
                outcome.observed_undeterminable,
            ]
        });

        assert_eq!(
            counts,
            [
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1],
            ]
        );
    }
}
