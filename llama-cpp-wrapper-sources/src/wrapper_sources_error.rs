use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum WrapperSourcesError {
    #[error("required command line argument <{name}> is missing")]
    MissingArgument { name: &'static str },
    #[error("compilation database file {path} could not be created: {source}")]
    Create {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("wrapper source list could not be written to {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("compilation database could not be written to {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}
