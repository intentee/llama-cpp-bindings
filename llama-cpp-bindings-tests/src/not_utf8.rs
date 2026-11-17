use anyhow::Context as _;
use anyhow::Result;
use std::str::Utf8Error;
use std::string::FromUtf8Error;

/// # Errors
///
/// Never fails in practice: a lone `0xff` byte is not UTF-8.
pub fn not_utf8() -> Result<Utf8Error> {
    String::from_utf8(vec![0xff])
        .err()
        .as_ref()
        .map(FromUtf8Error::utf8_error)
        .context("a lone 0xff byte is not UTF-8")
}
