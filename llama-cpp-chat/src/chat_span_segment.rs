use std::ops::Range;

use crate::chat_piece_role::ChatPieceRole;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatSpanSegment {
    pub bytes: Range<usize>,
    pub role: ChatPieceRole,
}
