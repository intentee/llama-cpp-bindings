use crate::token::LlamaToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
pub struct TokenOutsideVocabulary {
    pub token: LlamaToken,
    pub n_vocab: i32,
}
