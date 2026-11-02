use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

/// Which step of setting up a phase failed, as a test declares the failure it expects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseSetupFailureKind {
    ModelDownload,
    MmprojDownload,
    MmprojPathNotUnicode,
    ModelLoad,
    MmprojLoad,
}

impl Display for PhaseSetupFailureKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::ModelDownload => "ModelDownload",
            Self::MmprojDownload => "MmprojDownload",
            Self::MmprojPathNotUnicode => "MmprojPathNotUnicode",
            Self::ModelLoad => "ModelLoad",
            Self::MmprojLoad => "MmprojLoad",
        };

        formatter.write_str(name)
    }
}
