use std::collections::HashSet;
use std::ffi::CString;

use llama_cpp_bindings_sys::LLAMA_DRY_MAX_SEQ_BREAKERS;

use crate::SamplingError;

const DRY_SEQUENCE_BREAKER_MAX_BYTES: usize = 40;

fn validated_dry_sequence_breaker(
    index: usize,
    seq_breaker: &[u8],
) -> Result<CString, SamplingError> {
    if seq_breaker.is_empty() {
        return Err(SamplingError::DrySequenceBreakerEmpty { index });
    }

    if seq_breaker.len() > DRY_SEQUENCE_BREAKER_MAX_BYTES {
        return Err(SamplingError::DrySequenceBreakerTooLong {
            index,
            length: seq_breaker.len(),
            max_length: DRY_SEQUENCE_BREAKER_MAX_BYTES,
        });
    }

    CString::new(seq_breaker)
        .map_err(|source| SamplingError::DrySequenceBreakerContainsNul { index, source })
}

/// Validates DRY sequence breakers against the limits `llama_sampler_init_dry` enforces by
/// silently skipping, truncating, or ignoring breakers.
///
/// # Errors
///
/// Returns the [`SamplingError`] variant describing the first breaker llama.cpp would skip,
/// truncate, or reject, or [`SamplingError::TooManyDistinctDrySequenceBreakers`] when more
/// distinct breakers are given than llama.cpp uses.
pub fn dry_sequence_breakers(
    seq_breakers: impl IntoIterator<Item = impl AsRef<[u8]>>,
) -> Result<Vec<CString>, SamplingError> {
    let validated = seq_breakers
        .into_iter()
        .enumerate()
        .map(|(index, seq_breaker)| validated_dry_sequence_breaker(index, seq_breaker.as_ref()))
        .collect::<Result<Vec<CString>, SamplingError>>()?;
    let distinct = validated.iter().collect::<HashSet<&CString>>().len();
    let max = LLAMA_DRY_MAX_SEQ_BREAKERS as usize;

    if distinct > max {
        return Err(SamplingError::TooManyDistinctDrySequenceBreakers { distinct, max });
    }

    Ok(validated)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::dry_sequence_breakers;
    use crate::SamplingError;

    fn distinct_breakers(count: usize) -> Vec<String> {
        (0..count)
            .map(|breaker_index| format!("breaker{breaker_index}"))
            .collect()
    }

    #[test]
    fn breakers_within_every_limit_are_passed_through_in_order() {
        assert_eq!(
            dry_sequence_breakers(["\n", ":"]),
            Ok(vec![c"\n".to_owned(), c":".to_owned()])
        );
    }

    #[test]
    fn an_empty_breaker_is_refused_instead_of_skipped() {
        assert_eq!(
            dry_sequence_breakers(["\n", ""]),
            Err(SamplingError::DrySequenceBreakerEmpty { index: 1 })
        );
    }

    #[test]
    fn a_breaker_longer_than_forty_bytes_is_refused_instead_of_truncated() {
        assert_eq!(
            dry_sequence_breakers(["a".repeat(41)]),
            Err(SamplingError::DrySequenceBreakerTooLong {
                index: 0,
                length: 41,
                max_length: 40,
            })
        );
    }

    #[test]
    fn a_breaker_of_exactly_forty_bytes_is_accepted() {
        assert_eq!(
            dry_sequence_breakers(["a".repeat(40)]),
            Ok(vec![CString::new("a".repeat(40)).unwrap()])
        );
    }

    #[test]
    fn a_breaker_with_an_interior_nul_byte_is_refused() {
        let nul_error = CString::new("hello\0world").unwrap_err();

        assert_eq!(
            dry_sequence_breakers(["ok", "hello\0world"]),
            Err(SamplingError::DrySequenceBreakerContainsNul {
                index: 1,
                source: nul_error,
            })
        );
    }

    #[test]
    fn sixty_four_distinct_breakers_with_duplicates_are_accepted() {
        let mut breakers = distinct_breakers(64);
        breakers.push("breaker0".to_owned());

        assert_eq!(
            dry_sequence_breakers(&breakers).map(|validated| validated.len()),
            Ok(65)
        );
    }

    #[test]
    fn more_than_sixty_four_distinct_breakers_are_refused_instead_of_ignored() {
        assert_eq!(
            dry_sequence_breakers(distinct_breakers(65)),
            Err(SamplingError::TooManyDistinctDrySequenceBreakers {
                distinct: 65,
                max: 64,
            })
        );
    }
}
