use std::path::PathBuf;

use serde::Deserialize;

/// A reference from the file API's reply index to the reply file holding one object.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CmakeReplyFile {
    #[serde(rename = "jsonFile")]
    pub json_file: PathBuf,
}
