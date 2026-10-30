use std::mem;

use crate::ingest_outcome::IngestOutcome;
use crate::json_probe_outcome::JsonProbeOutcome;
use crate::sampled_token::SampledToken;
use crate::streaming_json_probe::StreamingJsonProbe;
use crate::token_piece::TokenPiece;

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProbeState {
    Idle,
    Holding {
        probe: StreamingJsonProbe,
        held: Vec<IngestOutcome>,
    },
}

const fn is_visible_content(outcome: &IngestOutcome) -> bool {
    matches!(outcome.sampled_token, SampledToken::Content(_))
        && matches!(outcome.piece, TokenPiece::Visible(_))
}

/// Holds visible content that opens a bare JSON object until the object settles.
///
/// A complete `{"name": ..., "arguments": ...}` object is released as tool-call tokens; anything
/// else is released as the content it was.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BareJsonToolCallProbe {
    state: ProbeState,
}

impl Default for BareJsonToolCallProbe {
    fn default() -> Self {
        Self {
            state: ProbeState::Idle,
        }
    }
}

impl BareJsonToolCallProbe {
    pub fn pass(&mut self, outcome: IngestOutcome, released: &mut Vec<IngestOutcome>) {
        let is_content = is_visible_content(&outcome);

        match mem::replace(&mut self.state, ProbeState::Idle) {
            ProbeState::Holding { held, .. } if !is_content => {
                released.extend(held);
                released.push(outcome);
            }
            ProbeState::Holding {
                mut probe,
                mut held,
            } => {
                let probe_outcome = probe.feed(outcome.piece.raw());

                held.push(outcome);
                self.settle(probe_outcome, probe, held, released);
            }
            ProbeState::Idle if is_content && outcome.piece.raw().trim_start().starts_with('{') => {
                let mut probe = StreamingJsonProbe::default();
                let probe_outcome = probe.feed(outcome.piece.raw());

                self.settle(probe_outcome, probe, vec![outcome], released);
            }
            ProbeState::Idle => released.push(outcome),
        }
    }

    fn settle(
        &mut self,
        probe_outcome: JsonProbeOutcome,
        probe: StreamingJsonProbe,
        held: Vec<IngestOutcome>,
        released: &mut Vec<IngestOutcome>,
    ) {
        match probe_outcome {
            JsonProbeOutcome::StillPossiblyValid => {
                self.state = ProbeState::Holding { probe, held };
            }
            JsonProbeOutcome::CompletedValid => {
                released.extend(held.into_iter().map(|outcome| IngestOutcome {
                    sampled_token: SampledToken::ToolCall(outcome.sampled_token.token()),
                    piece: outcome.piece,
                }));
            }
            JsonProbeOutcome::Failed => released.extend(held),
        }
    }

    /// Releases every held token as the content it was.
    pub fn release(&mut self, released: &mut Vec<IngestOutcome>) {
        if let ProbeState::Holding { held, .. } = mem::replace(&mut self.state, ProbeState::Idle) {
            released.extend(held);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BareJsonToolCallProbe;
    use crate::ingest_outcome::IngestOutcome;
    use crate::sampled_token::SampledToken;
    use crate::token::LlamaToken;
    use crate::token_piece::TokenPiece;

    fn content(piece: &str) -> IngestOutcome {
        IngestOutcome {
            sampled_token: SampledToken::Content(LlamaToken::new(1)),
            piece: TokenPiece::Visible(piece.to_owned()),
        }
    }

    fn tool_call(piece: &str) -> IngestOutcome {
        IngestOutcome {
            sampled_token: SampledToken::ToolCall(LlamaToken::new(1)),
            piece: TokenPiece::Visible(piece.to_owned()),
        }
    }

    fn pass_all(
        probe: &mut BareJsonToolCallProbe,
        outcomes: Vec<IngestOutcome>,
    ) -> Vec<IngestOutcome> {
        let mut released = Vec::new();

        for outcome in outcomes {
            probe.pass(outcome, &mut released);
        }

        released
    }

    #[test]
    fn content_that_does_not_open_an_object_passes_through() {
        let mut probe = BareJsonToolCallProbe::default();

        assert_eq!(pass_all(&mut probe, vec![content("Hi")]), [content("Hi")]);
    }

    #[test]
    fn a_complete_named_tool_call_object_is_released_as_tool_call_tokens() {
        let mut probe = BareJsonToolCallProbe::default();

        assert_eq!(
            pass_all(
                &mut probe,
                vec![
                    content(" {\"name\":"),
                    content("\"f\",\"arguments\""),
                    content(":{}}")
                ]
            ),
            [
                tool_call(" {\"name\":"),
                tool_call("\"f\",\"arguments\""),
                tool_call(":{}}"),
            ]
        );
    }

    #[test]
    fn an_object_that_is_not_a_tool_call_is_released_as_content() {
        let mut probe = BareJsonToolCallProbe::default();

        assert_eq!(
            pass_all(&mut probe, vec![content("{\"foo\":"), content("\"bar\"}")]),
            [content("{\"foo\":"), content("\"bar\"}")]
        );
    }

    #[test]
    fn an_unfinished_object_is_held_until_released() {
        let mut probe = BareJsonToolCallProbe::default();
        let mut released = pass_all(&mut probe, vec![content("{\"name\":")]);

        assert_eq!(released, []);

        probe.release(&mut released);

        assert_eq!(released, [content("{\"name\":")]);
    }

    #[test]
    fn a_token_outside_content_releases_the_held_object_before_itself() {
        let mut probe = BareJsonToolCallProbe::default();
        let reasoning = IngestOutcome {
            sampled_token: SampledToken::Reasoning(LlamaToken::new(2)),
            piece: TokenPiece::Visible("hmm".to_owned()),
        };

        assert_eq!(
            pass_all(&mut probe, vec![content("{\"name\":"), reasoning.clone()]),
            [content("{\"name\":"), reasoning]
        );
    }

    #[test]
    fn two_consecutive_tool_call_objects_are_recognized_separately() {
        let mut probe = BareJsonToolCallProbe::default();

        assert_eq!(
            pass_all(
                &mut probe,
                vec![
                    content("{\"name\":\"f\",\"arguments\":{}}"),
                    content("{\"name\":\"g\",\"arguments\":{}}"),
                ]
            ),
            [
                tool_call("{\"name\":\"f\",\"arguments\":{}}"),
                tool_call("{\"name\":\"g\",\"arguments\":{}}"),
            ]
        );
    }

    #[test]
    fn releasing_an_idle_probe_releases_nothing() {
        let mut released = Vec::new();

        BareJsonToolCallProbe::default().release(&mut released);

        assert_eq!(released, []);
    }
}
