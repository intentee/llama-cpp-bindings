use crate::token::LlamaToken;

/// A token of the model's vocabulary with the text it decodes to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VocabularyToken {
    pub token: LlamaToken,
    pub piece: String,
}
