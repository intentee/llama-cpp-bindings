use crate::marker_kind::MarkerKind;
use crate::token::LlamaToken;

/// Token sequences that end a section, one list per [`MarkerKind`].
///
/// A kind carries several alternatives because upstream templates can close a
/// reasoning block with more than one tag — for example `</think>` or a
/// `<tool_call>` that begins a tool call directly. An empty list means the kind
/// has no marker at all.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamingMarkers {
    pub reasoning_open: Vec<Vec<LlamaToken>>,
    pub reasoning_close: Vec<Vec<LlamaToken>>,
    pub tool_call_open: Vec<Vec<LlamaToken>>,
    pub tool_call_close: Vec<Vec<LlamaToken>>,
}

impl StreamingMarkers {
    #[must_use]
    pub const fn has_any(&self) -> bool {
        !self.reasoning_open.is_empty()
            || !self.reasoning_close.is_empty()
            || !self.tool_call_open.is_empty()
            || !self.tool_call_close.is_empty()
    }

    #[must_use]
    pub fn max_token_len(&self) -> usize {
        [
            &self.reasoning_open,
            &self.reasoning_close,
            &self.tool_call_open,
            &self.tool_call_close,
        ]
        .into_iter()
        .flatten()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
    }

    #[must_use]
    pub fn lookup(&self, kind: MarkerKind) -> &[Vec<LlamaToken>] {
        match kind {
            MarkerKind::ReasoningOpen => &self.reasoning_open,
            MarkerKind::ReasoningClose => &self.reasoning_close,
            MarkerKind::ToolCallOpen => &self.tool_call_open,
            MarkerKind::ToolCallClose => &self.tool_call_close,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::marker_kind::MarkerKind;

    use super::StreamingMarkers;
    use crate::token::LlamaToken;

    fn token(id: i32) -> LlamaToken {
        LlamaToken::new(id)
    }

    #[test]
    fn streaming_markers_with_no_markers_reports_none() {
        let markers = StreamingMarkers::default();
        assert!(!markers.has_any());
        assert_eq!(markers.max_token_len(), 0);
    }

    #[test]
    fn streaming_markers_max_token_len_takes_longest() {
        let markers = StreamingMarkers {
            reasoning_open: vec![vec![token(1)]],
            reasoning_close: vec![vec![token(2), token(3), token(4)]],
            tool_call_open: vec![vec![token(5), token(6)]],
            tool_call_close: Vec::new(),
        };
        assert_eq!(markers.max_token_len(), 3);
    }

    #[test]
    fn max_token_len_spans_every_alternative_of_one_kind() {
        let markers = StreamingMarkers {
            reasoning_close: vec![vec![token(1)], vec![token(2), token(3), token(4)]],
            ..StreamingMarkers::default()
        };

        assert_eq!(markers.max_token_len(), 3);
    }

    #[test]
    fn lookup_returns_every_alternative_for_a_kind() {
        let markers = StreamingMarkers {
            reasoning_close: vec![vec![token(1)], vec![token(2)]],
            ..StreamingMarkers::default()
        };

        assert_eq!(markers.lookup(MarkerKind::ReasoningClose).len(), 2);
        assert!(markers.lookup(MarkerKind::ToolCallOpen).is_empty());
    }
}
