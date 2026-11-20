use std::error::Error;

/// The message of `error` followed by the message of every error in its source chain.
#[must_use]
pub fn error_chain(error: &dyn Error) -> String {
    let mut chain = error.to_string();
    let mut source = error.source();

    while let Some(cause) = source {
        chain.push_str(": ");
        chain.push_str(&cause.to_string());
        source = cause.source();
    }

    chain
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::error_chain;

    #[test]
    fn an_error_without_a_source_is_its_own_message() {
        let not_utf8 = String::from_utf8(vec![0xff]).unwrap_err().utf8_error();

        assert_eq!(
            error_chain(&not_utf8),
            "invalid utf-8 sequence of 1 bytes from index 0"
        );
    }

    #[test]
    fn every_source_follows_the_message_it_caused() {
        let not_utf8 = CString::new(vec![0xff]).unwrap().into_string().unwrap_err();

        assert_eq!(
            error_chain(&not_utf8),
            "C string contained non-utf8 bytes: invalid utf-8 sequence of 1 bytes from index 0"
        );
    }
}
