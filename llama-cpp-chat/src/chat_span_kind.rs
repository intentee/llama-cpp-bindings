use llama_cpp_bindings_sys::bindings::llama_rs_chat_span_kind;
use llama_cpp_model::sampled_token_section::SampledTokenSection;

use crate::chat_piece_role::ChatPieceRole;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatSpanKind {
    ReasoningBlock,
    Reasoning,
    Content,
    ToolCall,
    ToolCallMarker,
}

impl ChatSpanKind {
    #[must_use]
    pub const fn piece_role(self) -> ChatPieceRole {
        match self {
            Self::ReasoningBlock => ChatPieceRole::Marker(SampledTokenSection::Reasoning),
            Self::Reasoning => ChatPieceRole::Text(SampledTokenSection::Reasoning),
            Self::Content => ChatPieceRole::Text(SampledTokenSection::Content),
            Self::ToolCall => ChatPieceRole::Text(SampledTokenSection::ToolCall),
            Self::ToolCallMarker => ChatPieceRole::Marker(SampledTokenSection::ToolCall),
        }
    }
}

impl From<llama_rs_chat_span_kind> for ChatSpanKind {
    fn from(kind: llama_rs_chat_span_kind) -> Self {
        match kind {
            llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_REASONING_BLOCK => {
                Self::ReasoningBlock
            }
            llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_REASONING => Self::Reasoning,
            llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_CONTENT => Self::Content,
            llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL => Self::ToolCall,
            llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER => {
                Self::ToolCallMarker
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_sys::bindings::llama_rs_chat_span_kind;
    use llama_cpp_model::sampled_token_section::SampledTokenSection;

    use super::ChatSpanKind;
    use crate::chat_piece_role::ChatPieceRole;

    #[test]
    fn text_spans_attribute_their_bytes_as_visible_text_of_their_section() {
        assert_eq!(
            [
                ChatSpanKind::Reasoning,
                ChatSpanKind::Content,
                ChatSpanKind::ToolCall
            ]
            .map(ChatSpanKind::piece_role),
            [
                ChatPieceRole::Text(SampledTokenSection::Reasoning),
                ChatPieceRole::Text(SampledTokenSection::Content),
                ChatPieceRole::Text(SampledTokenSection::ToolCall),
            ]
        );
    }

    #[test]
    fn marker_spans_attribute_their_bytes_as_markers_of_their_section() {
        assert_eq!(
            [ChatSpanKind::ReasoningBlock, ChatSpanKind::ToolCallMarker]
                .map(ChatSpanKind::piece_role),
            [
                ChatPieceRole::Marker(SampledTokenSection::Reasoning),
                ChatPieceRole::Marker(SampledTokenSection::ToolCall),
            ]
        );
    }

    #[test]
    fn every_span_kind_a_chat_reading_reports_maps_to_its_variant() {
        assert_eq!(
            [
                llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_REASONING_BLOCK,
                llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_REASONING,
                llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_CONTENT,
                llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL,
                llama_rs_chat_span_kind::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER,
            ]
            .map(ChatSpanKind::from),
            [
                ChatSpanKind::ReasoningBlock,
                ChatSpanKind::Reasoning,
                ChatSpanKind::Content,
                ChatSpanKind::ToolCall,
                ChatSpanKind::ToolCallMarker,
            ]
        );
    }
}
