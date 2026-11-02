use std::path::PathBuf;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cmake_cache::CmakeCache;

const BUNDLED_GGML_OPENMP_SWITCH: &str = "GGML_OPENMP_ENABLED";

const C_RUNTIME_LIBRARY_NAMES: &str = "OpenMP_C_LIB_NAMES";

const CXX_RUNTIME_LIBRARY_NAMES: &str = "OpenMP_CXX_LIB_NAMES";

fn bundled_ggml_uses_openmp(cache: &CmakeCache) -> Result<bool, BuildError> {
    match cache.value(BUNDLED_GGML_OPENMP_SWITCH)? {
        "ON" => Ok(true),
        "OFF" => Ok(false),
        value => Err(BuildError::CmakeCacheEntryNotASwitch {
            name: BUNDLED_GGML_OPENMP_SWITCH,
            value: value.to_owned(),
        }),
    }
}

fn runtime_library_names(cache: &CmakeCache) -> Result<Vec<&str>, BuildError> {
    let mut names = Vec::new();

    for library_names in [C_RUNTIME_LIBRARY_NAMES, CXX_RUNTIME_LIBRARY_NAMES] {
        for name in cache
            .value(library_names)?
            .split(';')
            .filter(|name| !name.is_empty())
        {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }

    Ok(names)
}

/// Where the build learns which OpenMP runtime libraries a statically linked ggml needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenmpRuntimeSource {
    SharedLibraries,
    BundledGgml,
    SystemGgml,
}

impl OpenmpRuntimeSource {
    #[must_use]
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Self {
        match (
            context.build_shared_libs,
            environment.has(BuildFeature::SystemGgml),
            environment.has(BuildFeature::SystemGgmlStatic),
        ) {
            (true, _, _) | (false, true, false) => Self::SharedLibraries,
            (false, true, true) => Self::SystemGgml,
            (false, false, _) => Self::BundledGgml,
        }
    }

    /// The OpenMP runtime libraries `CMake` resolved for ggml, which only a statically linked
    /// ggml leaves for the bindings to link.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::CmakeCacheEntryMissing`] when the cache lacks an entry that tells
    /// whether or where ggml's OpenMP runtime is, or [`BuildError::CmakeCacheEntryNotASwitch`]
    /// when the bundled ggml's OpenMP switch is neither `ON` nor `OFF`.
    pub fn libraries(self, cache: &CmakeCache) -> Result<Vec<PathBuf>, BuildError> {
        let uses_openmp = match self {
            Self::SharedLibraries => false,
            Self::BundledGgml => bundled_ggml_uses_openmp(cache)?,
            Self::SystemGgml => cache.contains(C_RUNTIME_LIBRARY_NAMES),
        };

        if !uses_openmp {
            return Ok(Vec::new());
        }

        runtime_library_names(cache)?
            .into_iter()
            .map(|name| {
                cache
                    .value(&format!("OpenMP_{name}_LIBRARY"))
                    .map(PathBuf::from)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use enumflags2::BitFlags;

    use super::OpenmpRuntimeSource;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::build_feature::BuildFeature;
    use crate::cmake_cache::tests::cache_of;

    const GNU_RUNTIME: [(&str, &str); 4] = [
        ("OpenMP_C_LIB_NAMES", "gomp;pthread"),
        ("OpenMP_CXX_LIB_NAMES", "gomp"),
        ("OpenMP_gomp_LIBRARY", "/usr/lib/gcc/libgomp.so"),
        ("OpenMP_pthread_LIBRARY", "/usr/lib/libpthread.a"),
    ];

    fn gnu_runtime_with(entry: (&'static str, &'static str)) -> Vec<(&'static str, &'static str)> {
        [&[entry][..], &GNU_RUNTIME].concat()
    }

    fn failure(source: OpenmpRuntimeSource, entries: &[(&str, &str)]) -> Option<String> {
        source
            .libraries(&cache_of(entries))
            .err()
            .map(|error| error.to_string())
    }

    #[test]
    fn only_a_statically_linked_ggml_leaves_its_runtime_to_the_bindings() {
        assert_eq!(
            [
                BitFlags::from(BuildFeature::DynamicLink),
                BitFlags::from(BuildFeature::SystemGgml),
                BuildFeature::SystemGgml | BuildFeature::SystemGgmlStatic,
                BitFlags::empty(),
            ]
            .map(|features| {
                let environment = cargo_environment(features, &[]);

                OpenmpRuntimeSource::of(&context_of(&environment), &environment)
            }),
            [
                OpenmpRuntimeSource::SharedLibraries,
                OpenmpRuntimeSource::SharedLibraries,
                OpenmpRuntimeSource::SystemGgml,
                OpenmpRuntimeSource::BundledGgml,
            ]
        );
    }

    #[test]
    fn shared_libraries_need_no_runtime_from_the_cache() {
        assert_eq!(
            OpenmpRuntimeSource::SharedLibraries
                .libraries(&cache_of(&[]))
                .unwrap(),
            Vec::<PathBuf>::new()
        );
    }

    #[test]
    fn a_bundled_ggml_with_openmp_needs_every_runtime_library_of_both_languages_once() {
        assert_eq!(
            OpenmpRuntimeSource::BundledGgml
                .libraries(&cache_of(&gnu_runtime_with(("GGML_OPENMP_ENABLED", "ON"))))
                .unwrap(),
            [
                PathBuf::from("/usr/lib/gcc/libgomp.so"),
                PathBuf::from("/usr/lib/libpthread.a")
            ]
        );
    }

    #[test]
    fn a_bundled_ggml_without_openmp_needs_no_runtime() {
        assert_eq!(
            OpenmpRuntimeSource::BundledGgml
                .libraries(&cache_of(&gnu_runtime_with(("GGML_OPENMP_ENABLED", "OFF"))))
                .unwrap(),
            Vec::<PathBuf>::new()
        );
    }

    #[test]
    fn a_bundled_ggml_openmp_switch_other_than_on_or_off_is_refused() {
        assert_eq!(
            failure(
                OpenmpRuntimeSource::BundledGgml,
                &gnu_runtime_with(("GGML_OPENMP_ENABLED", "TRUE"))
            ),
            Some(
                "the CMake cache entry GGML_OPENMP_ENABLED is TRUE, neither ON nor OFF".to_owned()
            )
        );
    }

    #[test]
    fn a_system_ggml_needs_the_runtime_only_when_its_config_searched_for_openmp() {
        assert_eq!(
            [cache_of(&[]), cache_of(&GNU_RUNTIME)]
                .map(|cache| OpenmpRuntimeSource::SystemGgml.libraries(&cache).unwrap()),
            [
                Vec::new(),
                vec![
                    PathBuf::from("/usr/lib/gcc/libgomp.so"),
                    PathBuf::from("/usr/lib/libpthread.a")
                ]
            ]
        );
    }

    #[test]
    fn a_bundled_ggml_with_openmp_but_no_runtime_library_list_is_refused() {
        assert_eq!(
            failure(
                OpenmpRuntimeSource::BundledGgml,
                &[("GGML_OPENMP_ENABLED", "ON")]
            ),
            Some("the CMake cache has no entry OpenMP_C_LIB_NAMES".to_owned())
        );
    }

    #[test]
    fn a_runtime_library_cmake_did_not_locate_is_refused() {
        assert_eq!(
            failure(
                OpenmpRuntimeSource::BundledGgml,
                &[
                    ("GGML_OPENMP_ENABLED", "ON"),
                    ("OpenMP_C_LIB_NAMES", "gomp"),
                    ("OpenMP_CXX_LIB_NAMES", "gomp"),
                ]
            ),
            Some("the CMake cache has no entry OpenMP_gomp_LIBRARY".to_owned())
        );
    }
}
