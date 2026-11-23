use crate::privatized_field::PrivatizedField;

/// The initial-cache script that, passed to `CMake` with `-C`, has every configure of the build
/// tree query its cache before `CMake` reads the queries.
pub const CACHE_QUERY_SCRIPT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/cmake/cache_query.cmake");

/// The directory of the native package that holds the llama.cpp sources it builds.
pub const LLAMA_CPP_DIR: &str = "llama.cpp";

/// The key under which the native package hands its llama.cpp source directory to the build
/// scripts of the packages that depend on it.
pub const LLAMA_CPP_ROOT_METADATA_KEY: &str = "root";

/// The variable cargo sets from [`LLAMA_CPP_ROOT_METADATA_KEY`] for the build scripts of packages
/// that depend on the native package, which links `llama`.
pub const LLAMA_CPP_ROOT_VARIABLE: &str = "DEP_LLAMA_ROOT";

pub const NATIVE_PROFILE: &str = "Release";

pub const PRIVATIZED_FIELDS: &[PrivatizedField] = &[PrivatizedField {
    type_name: "mtmd_context_params",
    field_name: "image_marker",
}];
