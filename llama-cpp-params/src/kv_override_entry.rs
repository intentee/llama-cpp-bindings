use std::ffi::CString;

use crate::param_override_value::ParamOverrideValue;

#[derive(Clone, Debug, PartialEq)]
pub struct KvOverrideEntry {
    pub key: CString,
    pub value: ParamOverrideValue,
}
