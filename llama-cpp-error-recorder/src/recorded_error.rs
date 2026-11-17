use std::error::Error;

#[derive(Debug, thiserror::Error)]
pub enum RecordedError {
    #[error("{message}")]
    FfiCallbackFailed {
        message: String,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
}

impl RecordedError {
    pub fn new(error: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        let source = error.into();

        Self::FfiCallbackFailed {
            message: source.to_string(),
            source,
        }
    }

    #[must_use]
    pub fn into_message(self) -> String {
        let Self::FfiCallbackFailed { message, .. } = self;

        message
    }

    #[must_use]
    pub fn message(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::RecordedError;

    #[test]
    fn message_returns_the_underlying_display() {
        let error = RecordedError::new("compute mask failed".to_string());

        assert_eq!(error.message(), "compute mask failed");
    }

    #[test]
    fn into_message_consumes_and_returns_the_display() {
        let error = RecordedError::new("reset failed".to_string());

        assert_eq!(error.into_message(), "reset failed");
    }

    #[test]
    fn display_formats_the_underlying_error() {
        let error = RecordedError::new("consume failed".to_string());

        assert_eq!(format!("{error}"), "consume failed");
    }

    #[test]
    fn source_exposes_the_underlying_error() {
        let error = RecordedError::new("inner boom".to_string());

        assert!(
            error
                .source()
                .is_some_and(|source| source.to_string() == "inner boom"),
            "a recorded error must expose its underlying error as the source"
        );
    }
}
