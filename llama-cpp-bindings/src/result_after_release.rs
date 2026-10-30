use std::fmt::Display;

/// Combines the result of reading a llama.cpp handle with the result of releasing it: a read
/// failure wins and a release failure that accompanies it is logged, since only one error can be
/// returned.
///
/// # Errors
///
/// Returns the read failure, or the release failure after a successful read.
pub fn result_after_release<TValue, TError: Display>(
    read: Result<TValue, TError>,
    released: Result<(), TError>,
) -> Result<TValue, TError> {
    match read {
        Ok(value) => released.map(|()| value),
        Err(read_failure) => {
            if let Err(release_failure) = released {
                log::error!("{release_failure}");
            }

            Err(read_failure)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::result_after_release;

    #[test]
    fn a_successful_read_and_release_yields_the_value() {
        assert_eq!(result_after_release::<u8, String>(Ok(7), Ok(())), Ok(7));
    }

    #[test]
    fn a_failed_release_after_a_successful_read_is_returned() {
        assert_eq!(
            result_after_release::<u8, String>(Ok(7), Err("release".to_owned())),
            Err("release".to_owned())
        );
    }

    #[test]
    fn a_failed_read_wins_over_a_failed_release() {
        assert_eq!(
            result_after_release::<u8, String>(Err("read".to_owned()), Err("release".to_owned())),
            Err("read".to_owned())
        );
    }
}
