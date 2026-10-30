use std::ffi::OsString;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CmakeDefine {
    pub name: &'static str,
    pub value: OsString,
}

impl CmakeDefine {
    #[must_use]
    pub fn new(name: &'static str, value: impl Into<OsString>) -> Self {
        Self {
            name,
            value: value.into(),
        }
    }

    #[must_use]
    pub fn switch(name: &'static str, enabled: bool) -> Self {
        Self::new(name, if enabled { "ON" } else { "OFF" })
    }
}
