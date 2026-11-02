use serde::Deserialize;

use crate::cmake_reply_file::CmakeReplyFile;

/// The replies `CMake` wrote for the shared stateless queries this crate submits.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CmakeReplies {
    #[serde(rename = "cache-v2")]
    pub cache: CmakeReplyFile,
}
