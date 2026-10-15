use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;
use crate::llama_token_attrs_from_int_error::LlamaTokenAttrsFromIntError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenAttrError {
    #[error(transparent)]
    TokenOutsideVocabulary(#[from] TokenOutsideVocabulary),
    #[error(transparent)]
    UnknownAttributes(#[from] LlamaTokenAttrsFromIntError),
}
