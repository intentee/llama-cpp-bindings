use std::ffi::CString;
use std::ffi::c_int;

#[derive(Debug, Eq, PartialEq)]
pub struct TokenizerInput {
    pub text: CString,
    pub length: c_int,
}
