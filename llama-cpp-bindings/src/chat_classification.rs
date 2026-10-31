use std::ops::Range;

use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_piece_role::ChatPieceRole;
use crate::chat_spans::ChatSpans;
use crate::sampled_token_section::SampledTokenSection;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatClassification {
    Spans(ChatSpans),
    OutputDoesNotMatchFormat(ChatOutputCompleteness),
}

impl ChatClassification {
    /// Returns how the token whose text occupies `bytes` of the output is attributed, or `None`
    /// while the classification cannot settle it yet. An answer that does not match the
    /// template's format attributes no token to a section once it is finished.
    #[must_use]
    pub fn classify_token(&self, bytes: &Range<usize>) -> Option<ChatPieceRole> {
        match self {
            Self::Spans(spans) => spans.classify_token(bytes),
            Self::OutputDoesNotMatchFormat(ChatOutputCompleteness::Partial) => None,
            Self::OutputDoesNotMatchFormat(ChatOutputCompleteness::Finished) => {
                Some(ChatPieceRole::Text(SampledTokenSection::Pending))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_classification::ChatClassification;
    use crate::chat_output_completeness::ChatOutputCompleteness;
    use crate::chat_piece_role::ChatPieceRole;
    use crate::sampled_token_section::SampledTokenSection;

    #[test]
    fn a_partial_answer_that_does_not_match_the_format_settles_no_token() {
        assert_eq!(
            ChatClassification::OutputDoesNotMatchFormat(ChatOutputCompleteness::Partial)
                .classify_token(&(0..3)),
            None
        );
    }

    #[test]
    fn a_finished_answer_that_does_not_match_the_format_is_visible_text_of_no_section() {
        assert_eq!(
            ChatClassification::OutputDoesNotMatchFormat(ChatOutputCompleteness::Finished)
                .classify_token(&(0..3)),
            Some(ChatPieceRole::Text(SampledTokenSection::Pending))
        );
    }
}
