use std::fmt::Debug;

use crate::model::params::LlamaModelParams;
use crate::model::params::kv_override_value_iterator::KvOverrideValueIterator;

#[derive(Debug)]
pub struct KvOverrides<'model_params> {
    model_params: &'model_params LlamaModelParams,
}

impl KvOverrides<'_> {
    #[must_use]
    pub const fn new(model_params: &LlamaModelParams) -> KvOverrides<'_> {
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

#[cfg(test)]
mod tests {
    use crate::model::params::kv_override_entry::KvOverrideEntry;
    use std::ffi::CString;
    use std::pin::pin;

    use crate::model::params::LlamaModelParams;
    use crate::model::params::param_override_value::ParamOverrideValue;

    #[test]
    fn kv_overrides_empty_by_default() {
        let params = LlamaModelParams::default();
        let overrides = params.kv_overrides();
        let count = overrides.into_iter().count();

        assert_eq!(count, 0);
    }

    #[test]
    fn kv_overrides_iterates_single_entry() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("test_key").unwrap();

        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(42))
            .unwrap();

        let entries: Result<Vec<_>, _> = params.kv_overrides().into_iter().collect();
        let entries = entries.expect("known override tags must convert");

        assert_eq!(
            entries,
            vec![KvOverrideEntry {
                key: CString::new("test_key").expect("the literal has no nul byte"),
                value: ParamOverrideValue::Int(42),
            }]
        );
    }

    #[test]
    fn kv_overrides_new_creates_view() {
        let params = LlamaModelParams::default();
        let overrides = super::KvOverrides::new(&params);
        let count = overrides.into_iter().count();

        assert_eq!(count, 0);
    }

    #[test]
    fn kv_overrides_preserves_unknown_tag_error() {
        let mut params = pin!(LlamaModelParams::default());
        let key = CString::new("valid_key").unwrap();

        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Int(99))
            .unwrap();

        params.kv_overrides[0].tag = 9999;

        let entry = params
            .kv_overrides()
            .into_iter()
            .next()
            .expect("one override must be present");

        assert_eq!(
            entry.unwrap_err(),
            crate::model::params::unknown_kv_override_tag::UnknownKvOverrideTag(9999)
        );
    }
}
