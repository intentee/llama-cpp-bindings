use crate::error::MarkerDetectionError;
use crate::marker_role::MarkerRole;
use crate::streaming_marker::StreamingMarker;
use crate::token::LlamaToken;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
/// The normalized streaming markers detected for a model.
pub struct StreamingMarkers {
    markers: Vec<StreamingMarker>,
}

impl StreamingMarkers {
    pub(crate) fn from_candidates(
        candidates: impl IntoIterator<Item = (Vec<LlamaToken>, MarkerRole)>,
    ) -> Result<Self, MarkerDetectionError> {
        let mut markers: Vec<StreamingMarker> = Vec::new();

        for (tokens, role) in candidates {
            if tokens.is_empty() {
                return Err(MarkerDetectionError::EmptyMarker);
            }

            if let Some(marker) = markers.iter_mut().find(|marker| marker.tokens() == tokens) {
                marker.add_role(role);
            } else {
                markers.push(StreamingMarker::new(tokens, role));
            }
        }

        if let Some(marker) = markers.iter().find(|marker| marker.opener_count() > 1) {
            return Err(MarkerDetectionError::AmbiguousMarkerOpeners {
                tokens: marker.tokens().to_vec(),
            });
        }

        Ok(Self { markers })
    }

    #[must_use]
    /// Returns whether the model exposes no streaming markers.
    pub const fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    #[must_use]
    /// Returns the number of distinct marker token sequences.
    pub const fn len(&self) -> usize {
        self.markers.len()
    }

    /// Iterates over the distinct marker token sequences.
    pub fn iter(&self) -> impl Iterator<Item = &StreamingMarker> {
        self.markers.iter()
    }

    #[must_use]
    pub fn max_token_len(&self) -> usize {
        self.markers
            .iter()
            .map(|marker| marker.tokens().len())
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn longest_matching_suffix(
        &self,
        tokens: &[LlamaToken],
    ) -> Option<&StreamingMarker> {
        self.markers
            .iter()
            .filter(|marker| tokens.ends_with(marker.tokens()))
            .max_by_key(|marker| marker.tokens().len())
    }

    pub(crate) fn is_prefix_of_longer_marker(&self, tokens: &[LlamaToken]) -> bool {
        self.markers.iter().any(|marker| {
            marker.tokens().len() > tokens.len() && marker.tokens().starts_with(tokens)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingMarkers;
    use crate::error::MarkerDetectionError;
    use crate::marker_role::MarkerRole;
    use crate::token::LlamaToken;

    fn token(id: i32) -> LlamaToken {
        LlamaToken::new(id)
    }

    #[test]
    fn empty_collection_reports_no_markers() {
        let markers = StreamingMarkers::default();

        assert!(markers.is_empty());
        assert_eq!(markers.len(), 0);
        assert_eq!(markers.max_token_len(), 0);
    }

    #[test]
    fn candidates_with_the_same_tokens_are_one_marker_with_multiple_roles() {
        let markers = StreamingMarkers::from_candidates([
            (vec![token(1)], MarkerRole::ReasoningClose),
            (vec![token(1)], MarkerRole::ToolCallOpen),
        ])
        .expect("a close and an opener compose into one transition");

        let marker = markers.iter().next().expect("one marker must remain");
        assert_eq!(marker.tokens(), &[token(1)]);
        assert_eq!(
            marker.roles(),
            &[MarkerRole::ReasoningClose, MarkerRole::ToolCallOpen]
        );
    }

    #[test]
    fn empty_marker_is_rejected() {
        assert_eq!(
            StreamingMarkers::from_candidates([(Vec::new(), MarkerRole::ReasoningOpen)]),
            Err(MarkerDetectionError::EmptyMarker)
        );
    }

    #[test]
    fn two_openers_for_the_same_tokens_are_rejected() {
        let marker_tokens = vec![token(1), token(2)];

        assert_eq!(
            StreamingMarkers::from_candidates([
                (marker_tokens.clone(), MarkerRole::ReasoningOpen),
                (marker_tokens.clone(), MarkerRole::ToolCallOpen),
            ]),
            Err(MarkerDetectionError::AmbiguousMarkerOpeners {
                tokens: marker_tokens
            })
        );
    }

    #[test]
    fn longest_matching_suffix_wins() {
        let markers = StreamingMarkers::from_candidates([
            (vec![token(2)], MarkerRole::ReasoningClose),
            (vec![token(1), token(2)], MarkerRole::ToolCallOpen),
        ])
        .expect("markers are valid");

        let matched = markers
            .longest_matching_suffix(&[token(1), token(2)])
            .expect("a suffix must match");

        assert_eq!(matched.tokens(), &[token(1), token(2)]);
        assert_eq!(matched.roles(), &[MarkerRole::ToolCallOpen]);
    }

    #[test]
    fn shorter_complete_marker_reports_when_it_is_still_an_ambiguous_prefix() {
        let markers = StreamingMarkers::from_candidates([
            (vec![token(1)], MarkerRole::ReasoningClose),
            (vec![token(1), token(2)], MarkerRole::ToolCallOpen),
        ])
        .expect("markers are valid");

        assert!(markers.is_prefix_of_longer_marker(&[token(1)]));
        assert!(!markers.is_prefix_of_longer_marker(&[token(1), token(2)]));
    }

    #[test]
    fn max_token_len_uses_the_longest_normalized_marker() {
        let markers = StreamingMarkers::from_candidates([
            (vec![token(1)], MarkerRole::ReasoningOpen),
            (
                vec![token(2), token(3), token(4)],
                MarkerRole::ReasoningClose,
            ),
            (vec![token(5), token(6)], MarkerRole::ToolCallOpen),
        ])
        .expect("markers are valid");

        assert_eq!(markers.max_token_len(), 3);
    }
}
