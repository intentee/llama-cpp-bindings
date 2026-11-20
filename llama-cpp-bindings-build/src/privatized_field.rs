/// A field of a C struct that the generated bindings keep private, so callers go through the
/// typed API instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivatizedField {
    pub type_name: &'static str,
    pub field_name: &'static str,
}
