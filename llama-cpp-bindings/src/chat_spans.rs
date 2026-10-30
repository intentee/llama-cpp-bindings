use std::cmp::Reverse;
use std::ops::Range;

use crate::chat_output_completeness::ChatOutputCompleteness;
use crate::chat_piece_role::ChatPieceRole;
use crate::chat_span::ChatSpan;
use crate::sampled_token_section::SampledTokenSection;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatSpans {
    pub parsed_length: usize,
    pub spans: Vec<ChatSpan>,
    pub completeness: ChatOutputCompleteness,
}

impl ChatSpans {
    fn tagged_role_at(&self, byte: usize) -> Option<ChatPieceRole> {
        self.spans
            .iter()
            .enumerate()
            .filter(|(_order, span)| span.bytes.contains(&byte))
            .min_by_key(|(order, span)| (span.bytes.len(), Reverse(*order)))
            .map(|(_order, span)| span.kind.piece_role())
    }

    fn untagged_marker_section(&self, byte: usize) -> Option<SampledTokenSection> {
        let next = (byte..self.parsed_length)
            .find_map(|later| self.tagged_role_at(later))
            .map(ChatPieceRole::section);

        if next.is_none() && self.completeness == ChatOutputCompleteness::Partial {
            return None;
        }

        let previous = (0..byte)
            .rev()
            .find_map(|earlier| self.tagged_role_at(earlier))
            .map(ChatPieceRole::section);

        Some(match (previous, next) {
            (Some(SampledTokenSection::Reasoning), _)
            | (_, Some(SampledTokenSection::Reasoning)) => SampledTokenSection::Reasoning,
            (Some(SampledTokenSection::ToolCall), _) | (_, Some(SampledTokenSection::ToolCall)) => {
                SampledTokenSection::ToolCall
            }
            _ => SampledTokenSection::Content,
        })
    }

    /// Returns how llama.cpp's parse attributes the token whose text occupies `bytes` of the
    /// output, or `None` while the parse has not settled the token yet.
    #[must_use]
    pub fn classify_token(&self, bytes: &Range<usize>) -> Option<ChatPieceRole> {
        if self.completeness == ChatOutputCompleteness::Partial && bytes.end > self.parsed_length {
            return None;
        }

        let roles = bytes
            .clone()
            .map(|byte| self.tagged_role_at(byte))
            .collect::<Vec<Option<ChatPieceRole>>>();

        if let Some(text_role) = roles
            .iter()
            .flatten()
            .find(|role| matches!(role, ChatPieceRole::Text(_)))
        {
            return Some(*text_role);
        }

        match roles.first() {
            Some(Some(marker_role)) => Some(*marker_role),
            Some(None) | None => self
                .untagged_marker_section(bytes.start)
                .map(ChatPieceRole::Marker),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_output_completeness::ChatOutputCompleteness;
    use crate::chat_piece_role::ChatPieceRole;
    use crate::chat_span::ChatSpan;
    use crate::chat_span_kind::ChatSpanKind;
    use crate::sampled_token_section::SampledTokenSection;

    use super::ChatSpans;

    fn span(kind: ChatSpanKind, start: usize, end: usize) -> ChatSpan {
        ChatSpan {
            kind,
            bytes: start..end,
        }
    }

    fn reasoning_then_content(completeness: ChatOutputCompleteness) -> ChatSpans {
        ChatSpans {
            parsed_length: 30,
            spans: vec![
                span(ChatSpanKind::Reasoning, 7, 12),
                span(ChatSpanKind::Content, 22, 30),
            ],
            completeness,
        }
    }

    #[test]
    fn a_token_inside_reasoning_text_is_visible_reasoning() {
        assert_eq!(
            reasoning_then_content(ChatOutputCompleteness::Partial).classify_token(&(7..12)),
            Some(ChatPieceRole::Text(SampledTokenSection::Reasoning))
        );
    }

    #[test]
    fn an_untagged_token_before_reasoning_opens_reasoning() {
        assert_eq!(
            reasoning_then_content(ChatOutputCompleteness::Partial).classify_token(&(0..7)),
            Some(ChatPieceRole::Marker(SampledTokenSection::Reasoning))
        );
    }

    #[test]
    fn an_untagged_token_after_reasoning_closes_reasoning() {
        assert_eq!(
            reasoning_then_content(ChatOutputCompleteness::Partial).classify_token(&(12..22)),
            Some(ChatPieceRole::Marker(SampledTokenSection::Reasoning))
        );
    }

    #[test]
    fn a_token_partly_in_content_text_is_visible_content() {
        assert_eq!(
            reasoning_then_content(ChatOutputCompleteness::Partial).classify_token(&(20..24)),
            Some(ChatPieceRole::Text(SampledTokenSection::Content))
        );
    }

    #[test]
    fn a_token_past_the_partial_parse_waits_for_more_output() {
        assert_eq!(
            reasoning_then_content(ChatOutputCompleteness::Partial).classify_token(&(28..31)),
            None
        );
    }

    #[test]
    fn the_innermost_span_decides_a_byte_inside_nested_spans() {
        let spans = ChatSpans {
            parsed_length: 20,
            spans: vec![
                span(ChatSpanKind::ToolCall, 0, 20),
                span(ChatSpanKind::ToolCallMarker, 0, 6),
            ],
            completeness: ChatOutputCompleteness::Partial,
        };

        assert_eq!(
            [
                spans.classify_token(&(0..6)),
                spans.classify_token(&(6..10))
            ],
            [
                Some(ChatPieceRole::Marker(SampledTokenSection::ToolCall)),
                Some(ChatPieceRole::Text(SampledTokenSection::ToolCall)),
            ]
        );
    }

    #[test]
    fn an_untagged_token_beside_a_tool_call_marks_the_tool_call() {
        let spans = ChatSpans {
            parsed_length: 20,
            spans: vec![
                span(ChatSpanKind::Content, 0, 5),
                span(ChatSpanKind::ToolCall, 10, 20),
            ],
            completeness: ChatOutputCompleteness::Partial,
        };

        assert_eq!(
            spans.classify_token(&(5..10)),
            Some(ChatPieceRole::Marker(SampledTokenSection::ToolCall))
        );
    }

    #[test]
    fn an_untagged_partial_tail_after_content_waits_for_what_follows() {
        let spans = ChatSpans {
            parsed_length: 8,
            spans: vec![span(ChatSpanKind::Content, 0, 5)],
            completeness: ChatOutputCompleteness::Partial,
        };

        assert_eq!(spans.classify_token(&(5..8)), None);
    }

    #[test]
    fn an_untagged_partial_tail_after_reasoning_waits_for_what_follows() {
        let spans = ChatSpans {
            parsed_length: 13,
            spans: vec![span(ChatSpanKind::Reasoning, 0, 12)],
            completeness: ChatOutputCompleteness::Partial,
        };

        assert_eq!(spans.classify_token(&(12..13)), None);
    }

    #[test]
    fn an_untagged_tail_of_a_finished_parse_after_content_is_a_content_marker() {
        let spans = ChatSpans {
            parsed_length: 8,
            spans: vec![span(ChatSpanKind::Content, 0, 5)],
            completeness: ChatOutputCompleteness::Finished,
        };

        assert_eq!(
            spans.classify_token(&(5..8)),
            Some(ChatPieceRole::Marker(SampledTokenSection::Content))
        );
    }

    #[test]
    fn a_token_without_text_takes_the_role_of_its_position() {
        assert_eq!(
            reasoning_then_content(ChatOutputCompleteness::Partial).classify_token(&(12..12)),
            Some(ChatPieceRole::Marker(SampledTokenSection::Reasoning))
        );
    }

    #[test]
    fn a_reasoning_block_byte_outside_reasoning_text_is_a_reasoning_marker() {
        let spans = ChatSpans {
            parsed_length: 20,
            spans: vec![
                span(ChatSpanKind::ReasoningBlock, 0, 20),
                span(ChatSpanKind::Reasoning, 7, 12),
            ],
            completeness: ChatOutputCompleteness::Partial,
        };

        assert_eq!(
            spans.classify_token(&(12..20)),
            Some(ChatPieceRole::Marker(SampledTokenSection::Reasoning))
        );
    }
}
