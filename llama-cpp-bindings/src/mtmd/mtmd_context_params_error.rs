use std::num::TryFromIntError;

use super::mtmd_image_token_limit::MtmdImageTokenLimit;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdContextParamsError {
    #[error("the image minimum token limit {limit:?} exceeds what a C int holds")]
    ImageMinTokensExceedCInt {
        limit: MtmdImageTokenLimit,
        #[source]
        source: TryFromIntError,
    },
    #[error("the image maximum token limit {limit:?} exceeds what a C int holds")]
    ImageMaxTokensExceedCInt {
        limit: MtmdImageTokenLimit,
        #[source]
        source: TryFromIntError,
    },
}
