use std::num::NonZeroU16;
use std::num::NonZeroU32;

use crate::batch_capacity_error::BatchCapacityError;

const MAX_TOKENS: u32 = i32::MAX.unsigned_abs();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchCapacity(NonZeroU32);

impl BatchCapacity {
    #[must_use]
    pub fn at_most(self, tokens: NonZeroU32) -> Self {
        Self(self.0.min(tokens))
    }

    #[must_use]
    pub const fn tokens(self) -> NonZeroU32 {
        self.0
    }

    #[must_use]
    pub const fn tokens_i32(self) -> i32 {
        self.0.get().cast_signed()
    }

    #[must_use]
    pub const fn tokens_usize(self) -> usize {
        self.0.get() as usize
    }
}

impl From<NonZeroU16> for BatchCapacity {
    fn from(tokens: NonZeroU16) -> Self {
        Self(NonZeroU32::from(tokens))
    }
}

impl TryFrom<u32> for BatchCapacity {
    type Error = BatchCapacityError;

    fn try_from(tokens: u32) -> Result<Self, Self::Error> {
        if tokens > MAX_TOKENS {
            return Err(BatchCapacityError::TooLarge {
                tokens,
                max_tokens: MAX_TOKENS,
            });
        }

        NonZeroU32::new(tokens)
            .map(Self)
            .ok_or(BatchCapacityError::Zero)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU16;
    use std::num::NonZeroU32;

    use super::BatchCapacity;
    use crate::batch_capacity_error::BatchCapacityError;

    #[test]
    fn rejects_a_batch_without_tokens() {
        assert_eq!(BatchCapacity::try_from(0), Err(BatchCapacityError::Zero));
    }

    #[test]
    fn rejects_a_batch_llama_cpp_cannot_address() {
        assert_eq!(
            BatchCapacity::try_from(u32::MAX),
            Err(BatchCapacityError::TooLarge {
                tokens: u32::MAX,
                max_tokens: i32::MAX.unsigned_abs(),
            })
        );
    }

    #[test]
    fn the_largest_batch_keeps_its_size_in_every_width() {
        let capacity = BatchCapacity::try_from(i32::MAX.unsigned_abs()).unwrap();

        assert_eq!(capacity.tokens().get(), i32::MAX.unsigned_abs());
        assert_eq!(capacity.tokens_i32(), i32::MAX);
        assert_eq!(capacity.tokens_usize(), i32::MAX.unsigned_abs() as usize);
    }

    #[test]
    fn a_batch_never_exceeds_the_token_bound_it_is_capped_at() {
        let capacity = BatchCapacity::try_from(2048).unwrap();

        assert_eq!(
            [
                capacity.at_most(NonZeroU32::new(512).unwrap()),
                capacity.at_most(NonZeroU32::MAX),
            ],
            [BatchCapacity::try_from(512).unwrap(), capacity]
        );
    }

    #[test]
    fn every_nonzero_sixteen_bit_size_is_a_valid_batch() {
        assert_eq!(
            BatchCapacity::from(NonZeroU16::MAX),
            BatchCapacity::try_from(u32::from(u16::MAX)).unwrap()
        );
    }
}
