use crate::token::LlamaToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
pub struct TokenOutsideVocabulary {
    pub token: LlamaToken,
    pub n_vocab: i32,
}

impl TokenOutsideVocabulary {
    /// # Errors
    ///
    /// Returns [`TokenOutsideVocabulary`] when `token` is not one of the `n_vocab` tokens of the
    /// vocabulary.
    pub fn check(token: LlamaToken, n_vocab: i32) -> Result<LlamaToken, Self> {
        if (0..n_vocab).contains(&token.0) {
            Ok(token)
        } else {
            Err(Self { token, n_vocab })
        }
    }
}
