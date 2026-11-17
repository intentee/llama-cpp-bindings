use anyhow::Context as _;
use anyhow::Result;
use std::str::Utf8Error;

/// # Errors
///
/// Never fails in practice: a lone `0xff` byte is not UTF-8.
pub fn not_utf8() -> Result<Utf8Error> {
    Ok(String::from_utf8(vec![0xff])
        .err()
        .context("a lone 0xff byte is not UTF-8")?
        .utf8_error())
}
