use crate::sampled_token::SampledToken;
use crate::token_piece::TokenPiece;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IngestOutcome {
    pub sampled_token: SampledToken,
    pub piece: TokenPiece,
}
