use std::ffi::CString;

use crate::model::params::param_override_value::ParamOverrideValue;

#[derive(Clone, Debug, PartialEq)]
pub struct KvOverrideEntry {
    pub key: CString,
    pub value: ParamOverrideValue,
}
