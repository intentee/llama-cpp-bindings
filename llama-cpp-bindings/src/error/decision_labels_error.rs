use std::str::Utf8Error;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecisionLabelsError {
    #[error("decision label {index} is not valid UTF-8")]
    LabelNotUtf8 {
        index: u32,
        #[source]
        source: Utf8Error,
    },
}
