use serde::Deserialize;

use crate::cmake_replies::CmakeReplies;

/// The file API's reply index, which names the reply file answering each query.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CmakeReplyIndex {
    pub reply: CmakeReplies,
}
