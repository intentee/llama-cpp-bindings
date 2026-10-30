use enumflags2::BitFlags;

use crate::build_feature::BuildFeature;

#[cfg(all(feature = "shared-stdcxx", feature = "static-stdcxx"))]
compile_error!("Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive");

/// The cargo features this build crate was compiled with.
#[must_use]
pub fn enabled_build_features() -> BitFlags<BuildFeature> {
    let mut features = BitFlags::empty();

    features.set(BuildFeature::Cuda, cfg!(feature = "cuda"));
    features.set(BuildFeature::CudaNoVmm, cfg!(feature = "cuda-no-vmm"));
    features.set(
        BuildFeature::DynamicBackends,
        cfg!(feature = "dynamic-backends"),
    );
    features.set(BuildFeature::DynamicLink, cfg!(feature = "dynamic-link"));
    features.set(
        BuildFeature::FaultInjection,
        cfg!(feature = "fault-injection"),
    );
    features.set(BuildFeature::Metal, cfg!(feature = "metal"));
    features.set(BuildFeature::Openmp, cfg!(feature = "openmp"));
    features.set(BuildFeature::Rocm, cfg!(feature = "rocm"));
    features.set(BuildFeature::SharedStdcxx, cfg!(feature = "shared-stdcxx"));
    features.set(BuildFeature::StaticStdcxx, cfg!(feature = "static-stdcxx"));
    features.set(BuildFeature::SystemGgml, cfg!(feature = "system-ggml"));
    features.set(
        BuildFeature::SystemGgmlStatic,
        cfg!(feature = "system-ggml-static"),
    );
    features.set(BuildFeature::Vulkan, cfg!(feature = "vulkan"));

    features
}

#[cfg(test)]
mod tests {
    use super::enabled_build_features;
    use crate::build_feature::BuildFeature;

    #[test]
    fn every_build_feature_mirrors_its_cargo_feature() {
        let features = enabled_build_features();

        assert_eq!(
            [
                features.contains(BuildFeature::Cuda),
                features.contains(BuildFeature::CudaNoVmm),
                features.contains(BuildFeature::DynamicBackends),
                features.contains(BuildFeature::DynamicLink),
                features.contains(BuildFeature::FaultInjection),
                features.contains(BuildFeature::Metal),
                features.contains(BuildFeature::Openmp),
                features.contains(BuildFeature::Rocm),
                features.contains(BuildFeature::SharedStdcxx),
                features.contains(BuildFeature::StaticStdcxx),
                features.contains(BuildFeature::SystemGgml),
                features.contains(BuildFeature::SystemGgmlStatic),
                features.contains(BuildFeature::Vulkan),
            ],
            [
                cfg!(feature = "cuda"),
                cfg!(feature = "cuda-no-vmm"),
                cfg!(feature = "dynamic-backends"),
                cfg!(feature = "dynamic-link"),
                cfg!(feature = "fault-injection"),
                cfg!(feature = "metal"),
                cfg!(feature = "openmp"),
                cfg!(feature = "rocm"),
                cfg!(feature = "shared-stdcxx"),
                cfg!(feature = "static-stdcxx"),
                cfg!(feature = "system-ggml"),
                cfg!(feature = "system-ggml-static"),
                cfg!(feature = "vulkan"),
            ]
        );
    }
}
