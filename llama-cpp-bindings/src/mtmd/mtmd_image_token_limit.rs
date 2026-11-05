use std::ffi::c_int;
use std::num::NonZeroU32;
use std::num::TryFromIntError;

/// A bound on the number of tokens the vision encoder produces for one image of a model with
/// dynamic resolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtmdImageTokenLimit {
    ModelMetadata,
    Tokens(NonZeroU32),
}

impl MtmdImageTokenLimit {
    #[must_use]
    pub fn from_c_int(raw: c_int) -> Self {
        u32::try_from(raw)
            .ok()
            .and_then(NonZeroU32::new)
            .map_or(Self::ModelMetadata, Self::Tokens)
    }

    /// # Errors
    ///
    /// Returns the conversion error when the token count exceeds what a C `int` holds.
    pub fn to_c_int(self) -> Result<c_int, TryFromIntError> {
        match self {
            Self::ModelMetadata => Ok(-1),
            Self::Tokens(tokens) => c_int::try_from(tokens.get()),
        }
    }

    #[must_use]
    pub fn at_most(self, ceiling: NonZeroU32) -> Self {
        match self {
            Self::ModelMetadata => Self::ModelMetadata,
            Self::Tokens(tokens) => Self::Tokens(tokens.min(ceiling)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::mtmd::mtmd_image_token_limit::MtmdImageTokenLimit;

    const SIXTY_FOUR: NonZeroU32 = NonZeroU32::new(64).unwrap();

    #[test]
    fn a_non_positive_c_value_reads_the_limit_from_the_model_metadata() {
        assert_eq!(
            [-1, 0].map(MtmdImageTokenLimit::from_c_int),
            [MtmdImageTokenLimit::ModelMetadata; 2]
        );
    }

    #[test]
    fn a_positive_c_value_is_a_token_limit() {
        assert_eq!(
            MtmdImageTokenLimit::from_c_int(64),
            MtmdImageTokenLimit::Tokens(SIXTY_FOUR)
        );
    }

    #[test]
    fn the_model_metadata_limit_is_the_c_value_minus_one() {
        assert_eq!(MtmdImageTokenLimit::ModelMetadata.to_c_int(), Ok(-1));
    }

    #[test]
    fn a_token_limit_is_its_c_value() {
        assert_eq!(MtmdImageTokenLimit::Tokens(SIXTY_FOUR).to_c_int(), Ok(64));
    }

    #[test]
    fn a_token_limit_beyond_a_c_int_does_not_convert() {
        assert_eq!(
            MtmdImageTokenLimit::Tokens(NonZeroU32::MAX).to_c_int(),
            i32::try_from(u32::MAX)
        );
    }

    #[test]
    fn a_token_limit_above_the_ceiling_drops_to_the_ceiling() {
        assert_eq!(
            MtmdImageTokenLimit::Tokens(NonZeroU32::MAX).at_most(SIXTY_FOUR),
            MtmdImageTokenLimit::Tokens(SIXTY_FOUR)
        );
    }

    #[test]
    fn the_model_metadata_limit_stays_under_a_ceiling() {
        assert_eq!(
            MtmdImageTokenLimit::ModelMetadata.at_most(SIXTY_FOUR),
            MtmdImageTokenLimit::ModelMetadata
        );
    }
}
