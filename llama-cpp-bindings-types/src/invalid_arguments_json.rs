use serde::Deserialize;
use serde::Serialize;

/// Why tool call arguments are not JSON: the parser's message and the 1-based line and column
/// where it stopped.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InvalidArgumentsJson {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl From<serde_json::Error> for InvalidArgumentsJson {
    fn from(error: serde_json::Error) -> Self {
        Self {
            message: error.to_string(),
            line: error.line(),
            column: error.column(),
        }
    }
}
