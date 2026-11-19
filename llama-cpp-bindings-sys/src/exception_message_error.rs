use std::string::FromUtf8Error;

/// An error type that can say a C++ exception message it was handed is not valid UTF-8.
pub trait ExceptionMessageError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self;
}
