use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq, serde::Serialize)]
pub struct CompileCommand {
    pub arguments: Vec<String>,
    pub directory: PathBuf,
    pub file: PathBuf,
}
