use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum WrapperSourcesError {
    #[error("required command line argument <{name}> is missing")]
    MissingArgument { name: &'static str },
    #[error("compilation database file {path} could not be created")]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("wrapper source list could not be written to {path}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("compilation database could not be written to {path}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}
