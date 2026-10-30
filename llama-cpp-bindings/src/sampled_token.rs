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
    use super::SampledToken;
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
}
