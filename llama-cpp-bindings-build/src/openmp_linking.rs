use std::path::PathBuf;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::cmake_cache::CmakeCache;
use crate::openmp_runtime_source::OpenmpRuntimeSource;
use crate::target_os::TargetOs;

fn link_by_file_name(library: PathBuf) -> Result<[CargoDirective; 2], BuildError> {
    match (library.parent(), library.file_name()) {
        (Some(directory), Some(file_name)) => Ok([
            CargoDirective::LinkSearchNative(directory.to_path_buf()),
            CargoDirective::LinkDylibByFileName(PathBuf::from(file_name)),
        ]),
        _ => Err(BuildError::OpenmpRuntimeLibraryWithoutFileName { library }),
    }
}

/// Links the OpenMP runtime `CMake` resolved for ggml, which is only known once `CMake` has
/// configured the build.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenmpLinking {
    pub source: OpenmpRuntimeSource,
    pub target_os: TargetOs,
}

impl OpenmpLinking {
    #[must_use]
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Self {
        Self {
            source: OpenmpRuntimeSource::of(context, environment),
            target_os: context.target_os.clone(),
        }
    }

    /// Links each runtime library by its exact file name from its own directory, after every
    /// library that needs it.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`OpenmpRuntimeSource::libraries`],
    /// [`BuildError::OpenmpRuntimeUnlinkableOnApple`] when an Apple target needs a runtime, or
    /// [`BuildError::OpenmpRuntimeLibraryWithoutFileName`] when a runtime library path names no
    /// file.
    pub fn directives(&self, cache: &CmakeCache) -> Result<Vec<CargoDirective>, BuildError> {
        let libraries = self.source.libraries(cache)?;

        if !libraries.is_empty() && matches!(self.target_os, TargetOs::Apple) {
            return Err(BuildError::OpenmpRuntimeUnlinkableOnApple { libraries });
        }

        libraries
            .into_iter()
            .map(link_by_file_name)
            .collect::<Result<Vec<[CargoDirective; 2]>, BuildError>>()
            .map(|directives| directives.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::OpenmpLinking;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_cache::tests::cache_of;
    use crate::openmp_runtime_source::OpenmpRuntimeSource;
    use crate::target_os::TargetOs;

    fn linux_directives(entries: &[(&str, &str)]) -> Result<Vec<CargoDirective>, String> {
        OpenmpLinking {
            source: OpenmpRuntimeSource::BundledGgml,
            target_os: TargetOs::Linux,
        }
        .directives(&cache_of(entries))
        .map_err(|error| error.to_string())
    }

    #[test]
    fn each_runtime_library_is_linked_by_its_exact_file_name_from_its_directory() {
        assert_eq!(
            linux_directives(&[
                ("GGML_OPENMP_ENABLED", "ON"),
                ("OpenMP_C_LIB_NAMES", "gomp;pthread"),
                ("OpenMP_CXX_LIB_NAMES", "gomp;pthread"),
                ("OpenMP_gomp_LIBRARY", "/usr/lib/gcc/libgomp.so"),
                ("OpenMP_pthread_LIBRARY", "/usr/lib/libpthread.a"),
            ]),
            Ok(vec![
                CargoDirective::LinkSearchNative(PathBuf::from("/usr/lib/gcc")),
                CargoDirective::LinkDylibByFileName(PathBuf::from("libgomp.so")),
                CargoDirective::LinkSearchNative(PathBuf::from("/usr/lib")),
                CargoDirective::LinkDylibByFileName(PathBuf::from("libpthread.a")),
            ])
        );
    }

    #[test]
    fn a_cache_without_the_bundled_ggml_openmp_switch_is_refused() {
        assert_eq!(
            linux_directives(&[]),
            Err("the CMake cache has no entry GGML_OPENMP_ENABLED".to_owned())
        );
    }

    #[test]
    fn a_runtime_library_path_without_a_file_name_is_refused() {
        assert_eq!(
            linux_directives(&[
                ("GGML_OPENMP_ENABLED", "ON"),
                ("OpenMP_C_LIB_NAMES", "gomp"),
                ("OpenMP_CXX_LIB_NAMES", "gomp"),
                ("OpenMP_gomp_LIBRARY", "/"),
            ]),
            Err("the OpenMP runtime library / names no file".to_owned())
        );
    }
}
