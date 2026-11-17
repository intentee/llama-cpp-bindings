use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::chat_output::ChatOutput;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_section::SampledTokenSection;

fn increment(observed: &mut u64) -> Result<()> {
    *observed = observed
        .checked_add(1)
        .context("the observed token count must fit u64")?;

    Ok(())
}

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
    /// # Errors
    ///
    /// Fails when the count of tokens observed in `section` does not fit `u64`.
    pub fn record_end_of_generation(&mut self, section: SampledTokenSection) -> Result<()> {
        self.eog_seen = true;

        let observed = match section {
            SampledTokenSection::Content => &mut self.observed_content,
            SampledTokenSection::Reasoning => &mut self.observed_reasoning,
            SampledTokenSection::ToolCall => &mut self.observed_tool_call,
            SampledTokenSection::Pending => &mut self.observed_undeterminable,
        };

        increment(observed)
    }

    /// # Errors
    ///
    /// Fails when the count of tokens observed in the token's section does not fit `u64`.
    pub fn record_outcome(&mut self, ingest: &IngestOutcome) -> Result<()> {
        self.generated
            .push(ingest.sampled_token.token(), ingest.piece.raw());

        match ingest.sampled_token {
            SampledToken::Content(_) => {
                self.content_stream.push_str(ingest.piece.visible());

                increment(&mut self.observed_content)
            }
            SampledToken::Reasoning(_) => {
                self.reasoning_stream.push_str(ingest.piece.visible());

                increment(&mut self.observed_reasoning)
            }
            SampledToken::ToolCall(_) => increment(&mut self.observed_tool_call),
            SampledToken::Undeterminable(_) => increment(&mut self.observed_undeterminable),
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings::ingest_outcome::IngestOutcome;
    use llama_cpp_bindings::sampled_token::SampledToken;
    use llama_cpp_bindings::sampled_token_section::SampledTokenSection;
    use llama_cpp_bindings::token::LlamaToken;
    use llama_cpp_bindings::token_piece::TokenPiece;

    use super::ClassifySampleLoopOutcome;

    #[test]
    fn records_a_tool_call_token_without_streaming_it() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome
            .record_outcome(&IngestOutcome {
                sampled_token: SampledToken::ToolCall(LlamaToken(42)),
                piece: TokenPiece::Visible("{".to_owned()),
            })
            .unwrap();

        assert_eq!(outcome.observed_tool_call, 1);
        assert_eq!(outcome.generated.text(), "{");
        assert_eq!(outcome.content_stream, "");
    }

    #[test]
    fn streams_the_visible_part_of_a_reasoning_token() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome
            .record_outcome(&IngestOutcome {
                sampled_token: SampledToken::Reasoning(LlamaToken(7)),
                piece: TokenPiece::Visible("thinking".to_owned()),
            })
            .unwrap();

        assert_eq!(outcome.observed_reasoning, 1);
        assert_eq!(outcome.reasoning_stream, "thinking");
    }

    #[test]
    fn counts_an_undeterminable_token_without_streaming_it() {
        let mut outcome = ClassifySampleLoopOutcome::default();

        outcome
            .record_outcome(&IngestOutcome {
                sampled_token: SampledToken::Undeterminable(LlamaToken(9)),
                piece: TokenPiece::Visible("ignored".to_owned()),
            })
            .unwrap();

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

            outcome.record_end_of_generation(section).unwrap();

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
