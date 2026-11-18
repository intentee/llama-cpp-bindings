use std::str::Utf8Error;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WrapperStringError {
    #[error(
        "{operation} returned an invalid FFI response: reported a thrown C++ exception without an \
         error message"
    )]
    ExceptionMessageMissing { operation: &'static str },
    #[error("{operation} reported text that is not valid UTF-8")]
    NotUtf8 {
        operation: &'static str,
        bytes: Vec<u8>,
        #[source]
        source: Utf8Error,
    },
}
