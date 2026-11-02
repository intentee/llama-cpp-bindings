use std::fs;
use std::path::PathBuf;

use cmake::Config;

use crate::build_error::BuildError;
use crate::cmake_define::CmakeDefine;
use crate::native_profile::NATIVE_PROFILE;

/// Configures the `CMake` project in `source_dir` and builds and installs it into `out_dir`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CmakeBuild {
    pub source_dir: PathBuf,
    pub out_dir: PathBuf,
    pub target_triple: String,
    pub host_triple: String,
    pub defines: Vec<CmakeDefine>,
    pub required_directories: Vec<PathBuf>,
}

impl CmakeBuild {
    /// # Errors
    ///
    /// Returns [`BuildError::Filesystem`] when a directory the project installs into cannot be
    /// created.
    ///
    /// # Panics
    ///
    /// Panics when `CMake` fails, the way the `cmake` crate reports a failed build.
    pub fn build(&self) -> Result<(), BuildError> {
        self.required_directories.iter().try_for_each(|directory| {
            fs::create_dir_all(directory).map_err(|source| BuildError::Filesystem {
                path: directory.clone(),
                source,
            })
        })?;

        let mut config = Config::new(&self.source_dir);

        for CmakeDefine { name, value } in &self.defines {
            config.define(name, value);
        }

        config
            .target(&self.target_triple)
            .host(&self.host_triple)
            .out_dir(&self.out_dir)
            .profile(NATIVE_PROFILE)
            .build();

        Ok(())
    }
}
