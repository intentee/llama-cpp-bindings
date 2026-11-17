use std::ops::Range;

use crate::chat_span_kind::ChatSpanKind;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatSpan {
    pub kind: ChatSpanKind,
    pub bytes: Range<usize>,
}
