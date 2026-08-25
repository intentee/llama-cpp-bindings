#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogRecord {
    pub level: log::Level,
    pub text: String,
}
