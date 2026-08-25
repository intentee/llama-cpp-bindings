use std::ffi::CString;

#[derive(Debug, Eq, PartialEq)]
pub struct SanitizedGrammar {
    pub grammar: CString,
    pub root: CString,
}
