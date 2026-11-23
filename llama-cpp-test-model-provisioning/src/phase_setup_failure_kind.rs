/// Which step of setting up a phase failed, as a test declares the failure it expects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseSetupFailureKind {
    ModelDownload,
    MmprojDownload,
    MmprojPathNotUnicode,
    ModelLoad,
    MmprojLoad,
}
