use crate::reasoning_markers::ReasoningMarkers;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbedReasoningMarkers {
    pub open: String,
    pub close: String,
}

impl From<ProbedReasoningMarkers> for ReasoningMarkers {
    fn from(ProbedReasoningMarkers { open, close }: ProbedReasoningMarkers) -> Self {
        Self {
            open,
            closes: vec![close],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::reasoning_markers::ReasoningMarkers;

    use super::ProbedReasoningMarkers;

    #[test]
    fn a_probed_pair_widens_to_a_single_candidate_list() {
        let widened: ReasoningMarkers = ProbedReasoningMarkers {
            open: "<think>".to_owned(),
            close: "</think>".to_owned(),
        }
        .into();

        assert_eq!(
            widened,
            ReasoningMarkers {
                open: "<think>".to_owned(),
                closes: vec!["</think>".to_owned()],
            }
        );
    }
}
