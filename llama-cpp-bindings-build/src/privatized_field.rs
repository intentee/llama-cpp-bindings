/// A field of a C struct that the generated bindings keep private, so callers go through the
/// typed API instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivatizedField {
    pub type_name: &'static str,
    pub field_name: &'static str,
}

impl PrivatizedField {
    pub const ALL: &'static [Self] = &[
        Self {
            type_name: "llama_context_params",
            field_name: "defrag_thold",
        },
        Self {
            type_name: "mtmd_context_params",
            field_name: "image_marker",
        },
    ];
}
