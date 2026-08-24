use crate::token::LlamaToken;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamingMarkers {
    pub reasoning_open: Option<Vec<LlamaToken>>,
    pub reasoning_closes: Vec<Vec<LlamaToken>>,
    pub tool_call_open: Option<Vec<LlamaToken>>,
    pub tool_call_close: Option<Vec<LlamaToken>>,
}

impl StreamingMarkers {
    #[must_use]
    pub const fn has_any(&self) -> bool {
        self.reasoning_open.is_some()
            || !self.reasoning_closes.is_empty()
            || self.tool_call_open.is_some()
            || self.tool_call_close.is_some()
    }

    #[must_use]
    pub fn max_token_len(&self) -> usize {
        [
            self.reasoning_open.as_deref(),
            self.tool_call_open.as_deref(),
            self.tool_call_close.as_deref(),
        ]
        .into_iter()
        .flatten()
        .map(<[LlamaToken]>::len)
        .chain(self.reasoning_closes.iter().map(Vec::len))
        .max()
        .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
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
            reasoning_open: Some(vec![token(1)]),
            reasoning_closes: vec![vec![token(2), token(3), token(4)]],
            tool_call_open: Some(vec![token(5), token(6)]),
            tool_call_close: None,
        };
        assert_eq!(markers.max_token_len(), 3);
    }
}
