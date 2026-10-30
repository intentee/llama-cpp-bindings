use crate::chat_spans::ChatSpans;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatClassification {
    Spans(ChatSpans),
    OutputDoesNotMatchFormat,
}
