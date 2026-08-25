use std::ffi::CStr;
use std::fmt::Debug;

use crate::model::params::LlamaModelParams;
use crate::model::params::kv_override_entry::KvOverrideEntry;
use crate::model::params::param_override_value::ParamOverrideValue;
use crate::model::params::unknown_kv_override_tag::UnknownKvOverrideTag;

#[derive(Debug)]
pub struct KvOverrideValueIterator<'model_params> {
    model_params: &'model_params LlamaModelParams,
    current: usize,
}

impl<'model_params> KvOverrideValueIterator<'model_params> {
    #[must_use]
    pub const fn new(model_params: &'model_params LlamaModelParams) -> Self {
        Self {
            model_params,
            current: 0,
        }
    }
}

impl Iterator for KvOverrideValueIterator<'_> {
    type Item = Result<KvOverrideEntry, UnknownKvOverrideTag>;

    fn next(&mut self) -> Option<Self::Item> {
        let overrides = self.model_params.params.kv_overrides;

        if overrides.is_null() {
            return None;
        }

        let current = unsafe { *overrides.add(self.current) };

        if current.key[0] == 0 {
            return None;
        }

        self.current += 1;
        let value = ParamOverrideValue::try_from(&current);

        Some(value.map(|value| KvOverrideEntry {
            key: unsafe { CStr::from_ptr(current.key.as_ptr()).to_owned() },
            value,
        }))
    }
}
