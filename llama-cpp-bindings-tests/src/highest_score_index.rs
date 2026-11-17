/// The index of the highest score, or `None` when there are no scores.
#[must_use]
pub fn highest_score_index(scores: &[f32]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::highest_score_index;

    #[test]
    fn the_highest_score_is_found_wherever_it_stands() {
        assert_eq!(highest_score_index(&[-1.5, 4.0, 0.25]), Some(1));
    }

    #[test]
    fn no_scores_have_no_highest_index() {
        assert_eq!(highest_score_index(&[]), None);
    }
}
