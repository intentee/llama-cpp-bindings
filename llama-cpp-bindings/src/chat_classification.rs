use std::ops::Range;

use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_piece_role::ChatPieceRole;
use crate::chat_spans::ChatSpans;
use crate::raw_chat_message::RawChatMessage;
use crate::sampled_token_section::SampledTokenSection;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatClassification {
    Spans(ChatSpans),
    OutputDoesNotMatchFormat(RawChatMessage),
}

impl ChatClassification {
    /// Returns how the token whose text occupies `bytes` of the output is attributed, or `None`
    /// while the classification cannot settle it yet. An answer that does not match the
    /// template's format attributes no token to a section once it is finished.
    #[must_use]
    pub fn classify_token(&self, bytes: &Range<usize>) -> Option<ChatPieceRole> {
        match self {
            Self::Spans(spans) => spans.classify_token(bytes),
            Self::OutputDoesNotMatchFormat(RawChatMessage {
                completeness: ChatOutputCompleteness::Partial,
                ..
            }) => None,
            Self::OutputDoesNotMatchFormat(RawChatMessage {
                completeness: ChatOutputCompleteness::Finished,
                ..
            }) => Some(ChatPieceRole::Text(SampledTokenSection::Pending)),
        }
    }

    /// The section the output ends in; an answer that does not match the template's format
    /// ends in no section.
    #[must_use]
    pub fn trailing_section(&self) -> SampledTokenSection {
        match self {
            Self::Spans(spans) => spans.trailing_section(),
            Self::OutputDoesNotMatchFormat(_) => SampledTokenSection::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_classification::ChatClassification;
    use crate::chat_output_completeness::ChatOutputCompleteness;
    use crate::chat_piece_role::ChatPieceRole;
    use crate::chat_span::ChatSpan;
    use crate::chat_span_kind::ChatSpanKind;
    use crate::chat_spans::ChatSpans;
    use crate::raw_chat_message::RawChatMessage;
    use crate::sampled_token_section::SampledTokenSection;

    fn unrecognized(completeness: ChatOutputCompleteness) -> ChatClassification {
        ChatClassification::OutputDoesNotMatchFormat(RawChatMessage {
            text: "odd".to_owned(),
            completeness,
            expected_format: "peg-native".to_owned(),
        })
    }

    #[test]
    fn a_partial_answer_that_does_not_match_the_format_settles_no_token() {
        assert_eq!(
            unrecognized(ChatOutputCompleteness::Partial).classify_token(&(0..3)),
            None
        );
    }

    #[test]
    fn a_finished_answer_that_does_not_match_the_format_is_visible_text_of_no_section() {
        assert_eq!(
            unrecognized(ChatOutputCompleteness::Finished).classify_token(&(0..3)),
            Some(ChatPieceRole::Text(SampledTokenSection::Pending))
        );
    }

    #[test]
    fn an_answer_that_matches_the_format_ends_in_the_section_of_its_last_span() {
        assert_eq!(
            ChatClassification::Spans(ChatSpans::new(
                4,
                &[ChatSpan {
                    kind: ChatSpanKind::ToolCall,
                    bytes: 0..4,
                }],
                ChatOutputCompleteness::Partial,
            ))
            .trailing_section(),
            SampledTokenSection::ToolCall
        );
    }

    #[test]
    fn an_answer_that_does_not_match_the_format_ends_in_no_section() {
        assert_eq!(
            unrecognized(ChatOutputCompleteness::Finished).trailing_section(),
            SampledTokenSection::Pending
        );
    }
}
