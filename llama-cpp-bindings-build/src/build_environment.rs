use std::collections::BTreeMap;
use std::env;
use std::env::VarError;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::path::PathBuf;

use find_cuda_helper::find_cuda_lib_dirs;

use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;

/// Everything a build script reads from outside.
///
/// The environment variables also name the cargo features of the package the build script belongs
/// to; the CUDA library directories are the ones installed on the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildEnvironment {
    pub variables: BTreeMap<OsString, OsString>,
    pub cuda_library_dirs: Vec<PathBuf>,
}

impl BuildEnvironment {
    /// The environment of the running build script.
    #[must_use]
    pub fn of_process() -> Self {
        Self {
            variables: env::vars_os().collect(),
            cuda_library_dirs: find_cuda_lib_dirs(),
        }
    }

    #[must_use]
    pub fn has(&self, feature: BuildFeature) -> bool {
        self.variables
            .contains_key(OsStr::new(feature.cargo_variable()))
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

    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when the variable is missing or not Unicode.
    pub fn required(&self, name: &'static str) -> Result<String, BuildError> {
        self.var(name)
            .map_err(|source| BuildError::Environment { name, source })
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
}

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    use find_cuda_helper::find_cuda_lib_dirs;

    use llama_cpp_error_chain::error_chain::error_chain;

    use super::BuildEnvironment;
    use crate::build_feature::BuildFeature;

    #[derive(Clone, Debug)]
    pub struct EnvironmentVariable {
        pub name: &'static str,
        pub value: OsString,
    }

    #[must_use]
    pub fn variable<TValue: Into<OsString>>(
        name: &'static str,
        value: TValue,
    ) -> EnvironmentVariable {
        EnvironmentVariable {
            name,
            value: value.into(),
        }
    }

    #[must_use]
    pub fn environment_of(
        features: &[BuildFeature],
        variables: &[EnvironmentVariable],
    ) -> BuildEnvironment {
        let mut environment = BuildEnvironment {
            variables: BTreeMap::new(),
            cuda_library_dirs: Vec::new(),
        };

        for EnvironmentVariable { name, value } in features
            .iter()
            .map(|feature| variable(feature.cargo_variable(), "1"))
            .chain(variables.iter().cloned())
        {
            environment.variables.insert(OsString::from(name), value);
        }

        environment
    }

    impl BuildEnvironment {
        #[must_use]
        pub fn with_non_unicode(mut self, name: &str) -> Self {
            self.variables
                .insert(OsString::from(name), OsString::from_vec(vec![0xff]));

            self
        }
    }

    #[test]
    fn the_process_environment_holds_the_variables_features_and_cuda_lookup_of_the_process() {
        let environment = BuildEnvironment::of_process();

        assert_eq!(
            environment.var("CARGO_MANIFEST_DIR"),
            Ok(env!("CARGO_MANIFEST_DIR").to_owned())
        );
        assert_eq!(environment.cuda_library_dirs, find_cuda_lib_dirs());
    }

    #[test]
    fn a_unicode_variable_is_read() {
        let environment = environment_of(
            &[],
            &[variable(
                "TARGET",
                OsString::from("x86_64-unknown-linux-gnu"),
            )],
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
    fn a_missing_variable_fails_when_required_and_is_none_when_optional() {
        let environment = environment_of(&[], &[]);

        assert_eq!(environment.optional("HOST").ok(), Some(None));
        assert_eq!(
            environment
                .required("HOST")
                .err()
                .map(|error| error_chain(&error)),
            Some(
                "environment variable HOST could not be read: environment variable not found"
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_variable_that_is_not_unicode_is_refused_even_when_optional() {
        let environment = environment_of(&[], &[]).with_non_unicode("TARGET");

        assert_eq!(
            environment
                .optional("TARGET")
                .err()
                .map(|error| error_chain(&error)),
            Some(
                "environment variable TARGET could not be read: environment variable was not \
                 valid unicode: \"\\xFF\""
                    .to_owned()
            )
        );
    }

    #[test]
    fn features_are_reported_by_membership() {
        let environment = environment_of(&[BuildFeature::Cuda], &[]);

        assert_eq!(
            [
                environment.has(BuildFeature::Cuda),
                environment.has(BuildFeature::Metal)
            ],
            [true, false]
        );
    }
}
