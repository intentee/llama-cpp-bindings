use crate::marker_role::MarkerRole;
use crate::sampled_token_section::SampledTokenSection;
use crate::token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
/// A normalized token sequence and every semantic role attached to it.
pub struct StreamingMarker {
    tokens: Vec<LlamaToken>,
    roles: Vec<MarkerRole>,
}

impl StreamingMarker {
    #[must_use]
    pub fn new(tokens: Vec<LlamaToken>, role: MarkerRole) -> Self {
        Self {
            tokens,
            roles: vec![role],
        }
    }

    pub fn add_role(&mut self, role: MarkerRole) {
        if !self.roles.contains(&role) {
            self.roles.push(role);
        }
    }

    #[must_use]
    /// Returns the tokens that form this marker.
    pub fn tokens(&self) -> &[LlamaToken] {
        &self.tokens
    }

    #[must_use]
    /// Returns the transitions associated with this marker.
    pub fn roles(&self) -> &[MarkerRole] {
        &self.roles
    }

    #[must_use]
    pub fn opener_count(&self) -> usize {
        self.roles
            .iter()
            .filter(|role| role.opened_section().is_some())
            .count()
    }

    fn opened_section(&self) -> Option<SampledTokenSection> {
        self.roles.iter().find_map(|role| role.opened_section())
    }

    #[must_use]
    pub fn span_section(&self, current: SampledTokenSection) -> SampledTokenSection {
        self.opened_section().unwrap_or_else(|| {
            if self
                .roles
                .iter()
                .any(|role| role.closed_section() == Some(current))
            {
                current
            } else {
                SampledTokenSection::Content
            }
        })
    }

    #[must_use]
    pub fn next_section(&self) -> SampledTokenSection {
        self.opened_section()
            .unwrap_or(SampledTokenSection::Content)
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingMarker;
    use crate::marker_role::MarkerRole;
    use crate::token::LlamaToken;

    #[test]
    fn a_role_the_marker_already_carries_is_not_added_twice() {
        let mut marker = StreamingMarker::new(vec![LlamaToken::new(1)], MarkerRole::ReasoningClose);

        marker.add_role(MarkerRole::ToolCallOpen);
        marker.add_role(MarkerRole::ToolCallOpen);

        assert_eq!(
            marker.roles(),
            &[MarkerRole::ReasoningClose, MarkerRole::ToolCallOpen]
        );
    }
}
