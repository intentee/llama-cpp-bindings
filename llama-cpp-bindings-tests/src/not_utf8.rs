use std::str::Utf8Error;
use std::string::FromUtf8Error;

#[must_use]
pub fn not_utf8() -> Option<Utf8Error> {
    String::from_utf8(vec![0xff])
        .err()
        .as_ref()
        .map(FromUtf8Error::utf8_error)
}
