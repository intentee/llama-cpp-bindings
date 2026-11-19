use std::fmt::Debug;

use crate::kv_override_value_iterator::KvOverrideValueIterator;

#[derive(Debug)]
pub struct KvOverrides<'model_params> {
    model_params: &'model_params llama_cpp_bindings_sys::bindings::llama_model_params,
}

impl KvOverrides<'_> {
    #[must_use]
    pub const fn new(
        model_params: &llama_cpp_bindings_sys::bindings::llama_model_params,
    ) -> KvOverrides<'_> {
        KvOverrides { model_params }
    }
}

impl<'model_params> IntoIterator for KvOverrides<'model_params> {
    type Item = <KvOverrideValueIterator<'model_params> as Iterator>::Item;
    type IntoIter = KvOverrideValueIterator<'model_params>;

    fn into_iter(self) -> Self::IntoIter {
        KvOverrideValueIterator::new(self.model_params)
    }
}
