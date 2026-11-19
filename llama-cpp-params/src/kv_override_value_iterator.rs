use std::ffi::CStr;
use std::fmt::Debug;

use crate::kv_override_entry::KvOverrideEntry;
use crate::param_override_value::ParamOverrideValue;
use crate::params_error::ParamsError;

#[derive(Debug)]
pub struct KvOverrideValueIterator<'model_params> {
    model_params: &'model_params llama_cpp_bindings_sys::bindings::llama_model_params,
    current: usize,
}

impl<'model_params> KvOverrideValueIterator<'model_params> {
    #[must_use]
    pub const fn new(
        model_params: &'model_params llama_cpp_bindings_sys::bindings::llama_model_params,
    ) -> Self {
        Self {
            model_params,
            current: 0,
        }
    }
}

impl Iterator for KvOverrideValueIterator<'_> {
    type Item = Result<KvOverrideEntry, ParamsError>;

    fn next(&mut self) -> Option<Self::Item> {
        let overrides = self.model_params.kv_overrides;

        if overrides.is_null() {
            return None;
        }

        let current = unsafe { *overrides.add(self.current) };

        if current.key[0] == 0 {
            return None;
        }

        self.current = self.current.saturating_add(1);
        let value = ParamOverrideValue::try_from(&current);

        Some(value.map(|value| KvOverrideEntry {
            key: unsafe { CStr::from_ptr(current.key.as_ptr()).to_owned() },
            value,
        }))
    }
}
