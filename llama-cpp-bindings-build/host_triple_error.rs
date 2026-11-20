use std::env::VarError;
use std::io;

#[derive(Debug, thiserror::Error)]
pub enum HostTripleError {
    #[error("the host triple could not be written into OUT_DIR")]
    HostTripleUnwritable(#[source] io::Error),
    #[error("cargo set no readable OUT_DIR for the build script")]
    OutDirUnreadable(#[source] VarError),
    #[error("cargo set no readable TARGET for the build script")]
    TargetUnreadable(#[source] VarError),
}
