use crate::reasoning_close_match::ReasoningCloseMatch;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReasoningMarkers {
    pub open: String,
    pub closes: Vec<String>,
}

impl ReasoningMarkers {
    #[must_use]
    pub fn is_usable(&self) -> bool {
        !self.open.is_empty() && self.closes.iter().any(|close| !close.is_empty())
    }

    #[must_use]
    pub fn find_earliest_close(&self, haystack: &str) -> Option<ReasoningCloseMatch> {
        self.closes
            .iter()
            .filter(|close| !close.is_empty())
            .filter_map(|close| {
                haystack
                    .find(close.as_str())
                    .map(|offset| ReasoningCloseMatch {
                        offset,
                        length: close.len(),
                    })
            })
            .min_by_key(|candidate| (candidate.offset, usize::MAX - candidate.length))
    }
}

#[cfg(test)]
mod tests {
    use crate::reasoning_close_match::ReasoningCloseMatch;

    use super::ReasoningMarkers;

    fn markers(closes: &[&str]) -> ReasoningMarkers {
        ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: closes.iter().map(|close| (*close).to_owned()).collect(),
        }
    }

    #[test]
    fn earliest_close_wins_over_a_later_one() {
        let found = markers(&["</think>", "<tool_call>"])
            .find_earliest_close("reasoning<tool_call>more</think>");

        assert_eq!(
            found,
            Some(ReasoningCloseMatch {
                offset: 9,
                length: "<tool_call>".len()
            })
        );
    }

    #[test]
    fn longest_candidate_wins_at_the_same_offset() {
        let found = markers(&["</think", "</think>"]).find_earliest_close("abc</think>");

        assert_eq!(
            found,
            Some(ReasoningCloseMatch {
                offset: 3,
                length: "</think>".len()
            })
        );
    }

    #[test]
    fn absent_candidates_yield_no_match() {
        assert_eq!(
            markers(&["</think>"]).find_earliest_close("no marker here"),
            None
        );
    }

    #[test]
    fn empty_candidates_are_ignored() {
        assert_eq!(markers(&[""]).find_earliest_close("anything"), None);
    }

    #[test]
    fn is_usable_requires_an_open_and_a_non_empty_close() {
        assert!(markers(&["</think>"]).is_usable());
        assert!(!markers(&[""]).is_usable());
        assert!(
            !ReasoningMarkers {
                open: String::new(),
                closes: vec!["</think>".to_owned()],
            }
            .is_usable()
        );
    }
}
