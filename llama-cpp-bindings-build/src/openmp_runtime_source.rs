use std::path::PathBuf;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cmake_cache::CmakeCache;

const BUNDLED_GGML_OPENMP_SWITCH: &str = "GGML_OPENMP_ENABLED";

const C_RUNTIME_LIBRARY_NAMES: &str = "OpenMP_C_LIB_NAMES";

const CXX_RUNTIME_LIBRARY_NAMES: &str = "OpenMP_CXX_LIB_NAMES";

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

    fn needs_runtime(self, cache: &CmakeCache) -> Result<bool, BuildError> {
        match self {
            Self::SharedLibraries => Ok(false),
            Self::BundledGgml => cache.switch(BUNDLED_GGML_OPENMP_SWITCH),
            Self::SystemGgml => Ok(cache.contains(C_RUNTIME_LIBRARY_NAMES)),
        }
    }

    /// The OpenMP runtime libraries `CMake` resolved for ggml, which only a statically linked
    /// ggml leaves for the bindings to link.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`CmakeCache::switch`], [`CmakeCache::list`], or
    /// [`CmakeCache::value`] when the cache does not tell whether or where ggml's OpenMP
    /// runtime is.
    pub fn libraries(self, cache: &CmakeCache) -> Result<Vec<PathBuf>, BuildError> {
        if !self.needs_runtime(cache)? {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();

        for library_names in [C_RUNTIME_LIBRARY_NAMES, CXX_RUNTIME_LIBRARY_NAMES] {
            for name in cache.list(library_names)? {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }

        names
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
    use crate::cmake_cache::CmakeCache;
    use crate::cmake_cache::tests::entry;
    use crate::cmake_cache_entry::CmakeCacheEntry;

    fn gnu_runtime() -> Vec<CmakeCacheEntry> {
        vec![
            entry("OpenMP_C_LIB_NAMES", "gomp;pthread"),
            entry("OpenMP_CXX_LIB_NAMES", "gomp"),
            entry("OpenMP_gomp_LIBRARY", "/usr/lib/gcc/libgomp.so"),
            entry("OpenMP_pthread_LIBRARY", "/usr/lib/libpthread.a"),
        ]
    }

    fn bundled_ggml_with_openmp(switch: &str) -> CmakeCache {
        CmakeCache {
            entries: [vec![entry("GGML_OPENMP_ENABLED", switch)], gnu_runtime()].concat(),
        }
    }

    fn gnu_runtime_libraries() -> Vec<PathBuf> {
        vec![
            PathBuf::from("/usr/lib/gcc/libgomp.so"),
            PathBuf::from("/usr/lib/libpthread.a"),
        ]
    }

    fn bundled_ggml_failure(entries: Vec<CmakeCacheEntry>) -> Option<String> {
        OpenmpRuntimeSource::BundledGgml
            .libraries(&CmakeCache { entries })
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
                .libraries(&CmakeCache {
                    entries: Vec::new()
                })
                .unwrap(),
            Vec::<PathBuf>::new()
        );
    }

    #[test]
    fn a_bundled_ggml_with_openmp_needs_every_runtime_library_of_both_languages_once() {
        assert_eq!(
            OpenmpRuntimeSource::BundledGgml
                .libraries(&bundled_ggml_with_openmp("ON"))
                .unwrap(),
            gnu_runtime_libraries()
        );
    }

    #[test]
    fn a_bundled_ggml_without_openmp_needs_no_runtime() {
        assert_eq!(
            OpenmpRuntimeSource::BundledGgml
                .libraries(&bundled_ggml_with_openmp("OFF"))
                .unwrap(),
            Vec::<PathBuf>::new()
        );
    }

    #[test]
    fn a_system_ggml_needs_the_runtime_only_when_its_config_searched_for_openmp() {
        assert_eq!(
            [Vec::new(), gnu_runtime()].map(|entries| {
                OpenmpRuntimeSource::SystemGgml
                    .libraries(&CmakeCache { entries })
                    .unwrap()
            }),
            [Vec::new(), gnu_runtime_libraries()]
        );
    }

    #[test]
    fn a_bundled_ggml_with_openmp_but_no_runtime_library_list_is_refused() {
        assert_eq!(
            bundled_ggml_failure(vec![entry("GGML_OPENMP_ENABLED", "ON")]),
            Some("the CMake cache has no entry OpenMP_C_LIB_NAMES".to_owned())
        );
    }

    #[test]
    fn a_runtime_library_cmake_did_not_locate_is_refused() {
        assert_eq!(
            bundled_ggml_failure(vec![
                entry("GGML_OPENMP_ENABLED", "ON"),
                entry("OpenMP_C_LIB_NAMES", "gomp"),
                entry("OpenMP_CXX_LIB_NAMES", "gomp"),
            ]),
            Some("the CMake cache has no entry OpenMP_gomp_LIBRARY".to_owned())
        );
    }
}
