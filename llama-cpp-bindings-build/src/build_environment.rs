use std::collections::BTreeMap;
use std::env;
use std::env::VarError;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::path::PathBuf;

use enumflags2::BitFlags;
use find_cuda_helper::find_cuda_lib_dirs;

use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::enabled_build_features::enabled_build_features;

/// Everything the native build reads from outside: the environment variables, the cargo features
/// this crate was compiled with, and the CUDA library directories installed on the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildEnvironment {
    pub variables: BTreeMap<OsString, OsString>,
    pub features: BitFlags<BuildFeature>,
    pub cuda_library_dirs: Vec<PathBuf>,
}

impl BuildEnvironment {
    /// The environment of the running build script.
    #[must_use]
    pub fn of_process() -> Self {
        Self {
            variables: env::vars_os().collect(),
            features: enabled_build_features(),
            cuda_library_dirs: find_cuda_lib_dirs(),
        }
    }

    /// # Errors
    ///
    /// Returns [`VarError::NotPresent`] for a variable the environment lacks, or
    /// [`VarError::NotUnicode`] for one that is not Unicode.
    pub fn var(&self, name: &'static str) -> Result<String, VarError> {
        self.variables
            .get(OsStr::new(name))
            .cloned()
            .ok_or(VarError::NotPresent)?
            .into_string()
            .map_err(VarError::NotUnicode)
    }

    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when the variable is missing or not Unicode.
    pub fn required(&self, name: &'static str) -> Result<String, BuildError> {
        self.var(name)
            .map_err(|source| BuildError::Environment { name, source })
    }

    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when the variable is set but not Unicode.
    pub fn optional(&self, name: &'static str) -> Result<Option<String>, BuildError> {
        match self.var(name) {
            Ok(value) => Ok(Some(value)),
            Err(VarError::NotPresent) => Ok(None),
            Err(source) => Err(BuildError::Environment { name, source }),
        }
    }

    #[must_use]
    pub fn has(&self, feature: BuildFeature) -> bool {
        self.features.contains(feature)
    }
}

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    use enumflags2::BitFlags;
    use find_cuda_helper::find_cuda_lib_dirs;

    use super::BuildEnvironment;
    use crate::build_feature::BuildFeature;
    use crate::enabled_build_features::enabled_build_features;

    #[must_use]
    pub fn environment_of<const VARIABLES: usize>(
        features: BitFlags<BuildFeature>,
        variables: [(&str, OsString); VARIABLES],
    ) -> BuildEnvironment {
        BuildEnvironment {
            variables: variables
                .into_iter()
                .map(|(name, value)| (OsString::from(name), value))
                .collect::<BTreeMap<OsString, OsString>>(),
            features,
            cuda_library_dirs: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_non_unicode(mut environment: BuildEnvironment, name: &str) -> BuildEnvironment {
        environment
            .variables
            .insert(OsString::from(name), OsString::from_vec(vec![0xff]));

        environment
    }

    #[test]
    fn the_process_environment_holds_the_variables_features_and_cuda_lookup_of_the_process() {
        let environment = BuildEnvironment::of_process();

        assert_eq!(
            environment.var("CARGO_MANIFEST_DIR"),
            Ok(env!("CARGO_MANIFEST_DIR").to_owned())
        );
        assert_eq!(environment.features, enabled_build_features());
        assert_eq!(environment.cuda_library_dirs, find_cuda_lib_dirs());
    }

    #[test]
    fn a_unicode_variable_is_read() {
        let environment = environment_of(
            BitFlags::empty(),
            [("TARGET", OsString::from("x86_64-unknown-linux-gnu"))],
        );

        assert_eq!(
            [
                environment.required("TARGET").ok(),
                environment.optional("TARGET").ok().flatten()
            ],
            [(); 2].map(|()| Some("x86_64-unknown-linux-gnu".to_owned()))
        );
    }

    #[test]
    fn a_missing_variable_is_required_but_optional() {
        let environment = environment_of(BitFlags::empty(), []);

        assert_eq!(environment.optional("HOST").ok(), Some(None));
        assert_eq!(
            environment
                .required("HOST")
                .err()
                .map(|error| error.to_string()),
            Some(
                "environment variable HOST could not be read: environment variable not found"
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_variable_that_is_not_unicode_is_refused_even_when_optional() {
        let environment = with_non_unicode(environment_of(BitFlags::empty(), []), "TARGET");

        assert_eq!(
            environment
                .optional("TARGET")
                .err()
                .map(|error| error.to_string()),
            Some(
                "environment variable TARGET could not be read: environment variable was not \
                 valid unicode: \"\\xFF\""
                    .to_owned()
            )
        );
    }

    #[test]
    fn features_are_reported_by_membership() {
        let environment = environment_of(BitFlags::from(BuildFeature::Cuda), []);

        assert_eq!(
            [
                environment.has(BuildFeature::Cuda),
                environment.has(BuildFeature::Metal)
            ],
            [true, false]
        );
    }
}
