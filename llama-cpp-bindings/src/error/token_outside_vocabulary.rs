use crate::token::LlamaToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
pub struct TokenOutsideVocabulary {
    pub token: LlamaToken,
    pub n_vocab: i32,
}

impl TokenOutsideVocabulary {
    /// The error of `token` when it is not one of the `n_vocab` tokens of the vocabulary.
    #[must_use]
    pub fn of(token: LlamaToken, n_vocab: i32) -> Option<Self> {
        if (0..n_vocab).contains(&token.0) {
            None
        } else {
            Some(Self { token, n_vocab })
        }
    }
}
