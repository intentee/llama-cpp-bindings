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
