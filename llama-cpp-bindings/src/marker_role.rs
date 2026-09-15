use crate::sampled_token_section::SampledTokenSection;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// A semantic transition performed when a streaming marker is consumed.
pub enum MarkerRole {
    ReasoningOpen,
    ReasoningClose,
    ToolCallOpen,
    ToolCallClose,
}

impl MarkerRole {
    #[must_use]
    pub const fn opened_section(self) -> Option<SampledTokenSection> {
        match self {
            Self::ReasoningOpen => Some(SampledTokenSection::Reasoning),
            Self::ToolCallOpen => Some(SampledTokenSection::ToolCall),
            Self::ReasoningClose | Self::ToolCallClose => None,
        }
    }

    #[must_use]
    pub const fn closed_section(self) -> Option<SampledTokenSection> {
        match self {
            Self::ReasoningClose => Some(SampledTokenSection::Reasoning),
            Self::ToolCallClose => Some(SampledTokenSection::ToolCall),
            Self::ReasoningOpen | Self::ToolCallOpen => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MarkerRole;
    use crate::sampled_token_section::SampledTokenSection;

    struct RoleSection {
        role: MarkerRole,
        section: SampledTokenSection,
    }

    #[test]
    fn an_opening_role_opens_its_section_and_closes_nothing() {
        let cases = [
            RoleSection {
                role: MarkerRole::ReasoningOpen,
                section: SampledTokenSection::Reasoning,
            },
            RoleSection {
                role: MarkerRole::ToolCallOpen,
                section: SampledTokenSection::ToolCall,
            },
        ];

        for RoleSection { role, section } in cases {
            assert_eq!(role.opened_section(), Some(section));
            assert_eq!(role.closed_section(), None);
        }
    }

    #[test]
    fn a_closing_role_closes_its_section_and_opens_nothing() {
        let cases = [
            RoleSection {
                role: MarkerRole::ReasoningClose,
                section: SampledTokenSection::Reasoning,
            },
            RoleSection {
                role: MarkerRole::ToolCallClose,
                section: SampledTokenSection::ToolCall,
            },
        ];

        for RoleSection { role, section } in cases {
            assert_eq!(role.closed_section(), Some(section));
            assert_eq!(role.opened_section(), None);
        }
    }
}
