use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::ffi::OsString;

use enumflags2::BitFlags;

use crate::build_feature::BuildFeature;

/// The features of the package whose build script runs, read from the variables cargo sets for
/// each feature the package enables.
#[must_use]
pub fn enabled_build_features(variables: &BTreeMap<OsString, OsString>) -> BitFlags<BuildFeature> {
    BitFlags::<BuildFeature>::all()
        .iter()
        .filter(|feature| variables.contains_key(OsStr::new(feature.cargo_variable())))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    use enumflags2::BitFlags;

    use super::enabled_build_features;
    use crate::build_feature::BuildFeature;

    #[test]
    fn only_the_features_cargo_sets_a_variable_for_are_enabled() {
        let variables = [
            "CARGO_FEATURE_CUDA",
            "CARGO_FEATURE_OPENMP",
            "CARGO_FEATURE_UNKNOWN",
        ]
        .map(|name| (OsString::from(name), OsString::from("1")))
        .into_iter()
        .collect::<BTreeMap<OsString, OsString>>();

        assert_eq!(
            enabled_build_features(&variables),
            BitFlags::from(BuildFeature::Cuda) | BuildFeature::Openmp
        );
    }
}
