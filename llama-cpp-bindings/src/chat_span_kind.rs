use crate::chat_piece_role::ChatPieceRole;
use crate::sampled_token_section::SampledTokenSection;

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

impl TryFrom<llama_cpp_bindings_sys::llama_rs_chat_span_kind> for ChatSpanKind {
    type Error = llama_cpp_ffi_status::FfiStatusError;

    fn try_from(
        kind: llama_cpp_bindings_sys::llama_rs_chat_span_kind,
    ) -> Result<Self, Self::Error> {
        match kind {
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_REASONING_BLOCK => {
                Ok(Self::ReasoningBlock)
            }
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_REASONING => Ok(Self::Reasoning),
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_CONTENT => Ok(Self::Content),
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL => Ok(Self::ToolCall),
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER => {
                Ok(Self::ToolCallMarker)
            }
            other => Err(llama_cpp_ffi_status::FfiStatusError {
                operation: "llama_rs_chat_spans_kind",
                code: i64::from(other),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChatSpanKind;
    use crate::chat_piece_role::ChatPieceRole;
    use crate::sampled_token_section::SampledTokenSection;

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
    fn every_wrapper_span_kind_maps_to_its_variant() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_REASONING_BLOCK,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_REASONING,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_CONTENT,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER,
            ]
            .map(ChatSpanKind::try_from),
            [
                Ok(ChatSpanKind::ReasoningBlock),
                Ok(ChatSpanKind::Reasoning),
                Ok(ChatSpanKind::Content),
                Ok(ChatSpanKind::ToolCall),
                Ok(ChatSpanKind::ToolCallMarker),
            ]
        );
    }

    #[test]
    fn an_unknown_wrapper_span_kind_is_a_status_error() {
        assert_eq!(
            ChatSpanKind::try_from(99),
            Err(llama_cpp_ffi_status::FfiStatusError {
                operation: "llama_rs_chat_spans_kind",
                code: 99,
            })
        );
    }
}
