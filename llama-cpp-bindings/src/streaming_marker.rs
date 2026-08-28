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
    pub(crate) fn new(tokens: Vec<LlamaToken>, role: MarkerRole) -> Self {
        Self {
            tokens,
            roles: vec![role],
        }
    }

    pub(crate) fn add_role(&mut self, role: MarkerRole) {
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

    pub(crate) fn opener_count(&self) -> usize {
        self.roles
            .iter()
            .filter(|role| role.opened_section().is_some())
            .count()
    }

    fn opened_section(&self) -> Option<SampledTokenSection> {
        self.roles.iter().find_map(|role| role.opened_section())
    }

    pub(crate) fn span_section(&self, current: SampledTokenSection) -> SampledTokenSection {
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

    pub(crate) fn next_section(&self) -> SampledTokenSection {
        self.opened_section()
            .unwrap_or(SampledTokenSection::Content)
    }
}
