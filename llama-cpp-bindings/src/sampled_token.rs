use crate::sampled_token_section::SampledTokenSection;
use crate::token::LlamaToken;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SampledToken {
    Content(LlamaToken),
    Reasoning(LlamaToken),
    ToolCall(LlamaToken),
    Undeterminable(LlamaToken),
}

impl SampledToken {
    #[must_use]
    pub const fn in_section(section: SampledTokenSection, token: LlamaToken) -> Self {
        match section {
            SampledTokenSection::Reasoning => Self::Reasoning(token),
            SampledTokenSection::Content => Self::Content(token),
            SampledTokenSection::ToolCall => Self::ToolCall(token),
            SampledTokenSection::Pending => Self::Undeterminable(token),
        }
    }

    #[must_use]
    pub const fn section(&self) -> SampledTokenSection {
        match self {
            Self::Reasoning(_) => SampledTokenSection::Reasoning,
            Self::Content(_) => SampledTokenSection::Content,
            Self::ToolCall(_) => SampledTokenSection::ToolCall,
            Self::Undeterminable(_) => SampledTokenSection::Pending,
        }
    }

    #[must_use]
    pub const fn token(&self) -> LlamaToken {
        match *self {
            Self::Content(token)
            | Self::Reasoning(token)
            | Self::ToolCall(token)
            | Self::Undeterminable(token) => token,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sampled_token::SampledToken;
    use crate::sampled_token_section::SampledTokenSection;
    use crate::token::LlamaToken;

    #[test]
    fn every_section_carries_the_sampled_token() {
        let token = LlamaToken::new(42);

        assert_eq!(
            [
                SampledToken::Content(token),
                SampledToken::Reasoning(token),
                SampledToken::ToolCall(token),
                SampledToken::Undeterminable(token),
            ]
            .map(|sampled_token| sampled_token.token()),
            [token; 4]
        );
    }

    #[test]
    fn a_token_placed_in_a_section_reports_that_section() {
        let token = LlamaToken::new(7);
        let sections = [
            SampledTokenSection::Pending,
            SampledTokenSection::Content,
            SampledTokenSection::Reasoning,
            SampledTokenSection::ToolCall,
        ];

        assert_eq!(
            sections.map(|section| SampledToken::in_section(section, token).section()),
            sections
        );
    }

    #[test]
    fn a_token_of_no_section_is_undeterminable() {
        assert_eq!(
            SampledToken::in_section(SampledTokenSection::Pending, LlamaToken::new(7)),
            SampledToken::Undeterminable(LlamaToken::new(7))
        );
    }
}
