use std::ops::Range;

use crate::token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatOutputToken {
    pub token: LlamaToken,
    pub bytes: Range<usize>,
}
