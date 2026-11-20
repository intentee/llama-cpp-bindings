use crate::privatized_field::PrivatizedField;

/// The initial-cache script that, passed to `CMake` with `-C`, has every configure of the build
/// tree query its cache before `CMake` reads the queries.
pub const CACHE_QUERY_SCRIPT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/cmake/cache_query.cmake");

pub const NATIVE_PROFILE: &str = "Release";

pub const PRIVATIZED_FIELDS: &[PrivatizedField] = &[PrivatizedField {
    type_name: "mtmd_context_params",
    field_name: "image_marker",
}];
