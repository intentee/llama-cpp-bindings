use crate::sampled_token_section::SampledTokenSection;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatPieceRole {
    Text(SampledTokenSection),
    Marker(SampledTokenSection),
}

impl ChatPieceRole {
    #[must_use]
    pub const fn section(self) -> SampledTokenSection {
        match self {
            Self::Text(section) | Self::Marker(section) => section,
        }
    }
}
