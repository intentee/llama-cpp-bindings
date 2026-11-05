use std::cell::Cell;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use bindgen::FieldVisibilityKind;
use bindgen::callbacks::DeriveInfo;
use bindgen::callbacks::EnumVariantCustomBehavior;
use bindgen::callbacks::EnumVariantValue;
use bindgen::callbacks::FieldInfo;
use bindgen::callbacks::ParseCallbacks;

use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::privatized_field::PrivatizedField;

const COMPARED_STRUCTS_WITH_FLOATS: &[&str] = &["llama_logit_bias", "llama_token_data"];

const COMPARED_STRUCTS_WITHOUT_FLOATS: &[&str] = &["llama_rs_batch_entry"];

const SUPERSEDED_ENUM_VARIANTS: &[&str] = &["GGML_PREC_DEFAULT"];

fn derives_for(type_name: &str) -> Vec<String> {
    if COMPARED_STRUCTS_WITHOUT_FLOATS.contains(&type_name) {
        vec!["PartialEq".to_owned(), "Eq".to_owned()]
    } else if COMPARED_STRUCTS_WITH_FLOATS.contains(&type_name) {
        vec!["PartialEq".to_owned()]
    } else {
        Vec::new()
    }
}

fn variant_behavior_for(variant_name: &str) -> Option<EnumVariantCustomBehavior> {
    SUPERSEDED_ENUM_VARIANTS
        .contains(&variant_name)
        .then_some(EnumVariantCustomBehavior::Hide)
}

/// Observes bindgen while it parses: keeps the [`PrivatizedField::ALL`] fields private, hides
/// enum variants llama.cpp superseded, derives the equality the fields of each struct the bindings
/// compare allow, and records what the bindings depend on.
#[derive(Clone, Debug)]
pub struct BindingCallbacks {
    privatized_field_hits: Rc<[Cell<bool>]>,
    rebuild_triggers: Rc<RefCell<Vec<CargoDirective>>>,
}

impl Default for BindingCallbacks {
    fn default() -> Self {
        Self {
            privatized_field_hits: PrivatizedField::ALL
                .iter()
                .map(|_| Cell::new(false))
                .collect(),
            rebuild_triggers: Rc::default(),
        }
    }
}

impl BindingCallbacks {
    /// # Errors
    ///
    /// Returns [`BuildError::PrivatizedFieldMissing`] for the first privatized field bindgen
    /// never saw.
    pub fn verify_every_privatized_field_was_found(&self) -> Result<(), BuildError> {
        PrivatizedField::ALL
            .iter()
            .zip(self.privatized_field_hits.iter())
            .find(|(_, was_found)| !was_found.get())
            .map_or(Ok(()), |(field, _)| {
                Err(BuildError::PrivatizedFieldMissing {
                    type_name: field.type_name,
                    field_name: field.field_name,
                })
            })
    }

    /// The headers and environment variables bindgen read, as cargo rebuild triggers.
    #[must_use]
    pub fn take_rebuild_triggers(&self) -> Vec<CargoDirective> {
        self.rebuild_triggers.take()
    }
}

impl ParseCallbacks for BindingCallbacks {
    fn header_file(&self, filename: &str) {
        self.rebuild_triggers
            .borrow_mut()
            .push(CargoDirective::RerunIfChanged(PathBuf::from(filename)));
    }

    fn include_file(&self, filename: &str) {
        self.rebuild_triggers
            .borrow_mut()
            .push(CargoDirective::RerunIfChanged(PathBuf::from(filename)));
    }

    fn read_env_var(&self, key: &str) {
        self.rebuild_triggers
            .borrow_mut()
            .push(CargoDirective::RerunIfEnvChanged(key.to_owned()));
    }

    fn add_derives(&self, info: &DeriveInfo<'_>) -> Vec<String> {
        derives_for(info.name)
    }

    fn enum_variant_behavior(
        &self,
        _enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<EnumVariantCustomBehavior> {
        variant_behavior_for(original_variant_name)
    }

    fn field_visibility(&self, info: FieldInfo<'_>) -> Option<FieldVisibilityKind> {
        PrivatizedField::ALL
            .iter()
            .zip(self.privatized_field_hits.iter())
            .find(|(field, _)| {
                field.type_name == info.type_name && field.field_name == info.field_name
            })
            .map(|(_, was_found)| {
                was_found.set(true);

                FieldVisibilityKind::Private
            })
    }
}

#[cfg(test)]
mod tests {
    use bindgen::callbacks::EnumVariantCustomBehavior;

    use super::derives_for;
    use super::variant_behavior_for;

    #[test]
    fn only_superseded_enum_variants_are_hidden() {
        assert_eq!(
            ["GGML_PREC_DEFAULT", "GGML_PREC_UNDEFINED"].map(variant_behavior_for),
            [Some(EnumVariantCustomBehavior::Hide), None]
        );
    }

    #[test]
    fn the_structs_the_bindings_compare_derive_equality_their_fields_allow() {
        assert_eq!(
            [
                "llama_token_data",
                "llama_logit_bias",
                "llama_rs_batch_entry",
                "llama_sampler_i"
            ]
            .map(derives_for),
            [
                vec!["PartialEq".to_owned()],
                vec!["PartialEq".to_owned()],
                vec!["PartialEq".to_owned(), "Eq".to_owned()],
                Vec::new()
            ]
        );
    }
}
