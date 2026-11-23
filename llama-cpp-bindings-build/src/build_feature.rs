#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuildFeature {
    Cuda,
    CudaNoVmm,
    DynamicBackends,
    DynamicLink,
    FaultInjection,
    Metal,
    Openmp,
    Rocm,
    SharedStdcxx,
    StaticStdcxx,
    SystemGgml,
    SystemGgmlStatic,
    Vulkan,
}

impl BuildFeature {
    /// The variable cargo sets for the build script of a package that enables this feature.
    #[must_use]
    pub const fn cargo_variable(self) -> &'static str {
        match self {
            Self::Cuda => "CARGO_FEATURE_CUDA",
            Self::CudaNoVmm => "CARGO_FEATURE_CUDA_NO_VMM",
            Self::DynamicBackends => "CARGO_FEATURE_DYNAMIC_BACKENDS",
            Self::DynamicLink => "CARGO_FEATURE_DYNAMIC_LINK",
            Self::FaultInjection => "CARGO_FEATURE_FAULT_INJECTION",
            Self::Metal => "CARGO_FEATURE_METAL",
            Self::Openmp => "CARGO_FEATURE_OPENMP",
            Self::Rocm => "CARGO_FEATURE_ROCM",
            Self::SharedStdcxx => "CARGO_FEATURE_SHARED_STDCXX",
            Self::StaticStdcxx => "CARGO_FEATURE_STATIC_STDCXX",
            Self::SystemGgml => "CARGO_FEATURE_SYSTEM_GGML",
            Self::SystemGgmlStatic => "CARGO_FEATURE_SYSTEM_GGML_STATIC",
            Self::Vulkan => "CARGO_FEATURE_VULKAN",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BuildFeature;

    #[test]
    fn every_feature_is_read_from_the_variable_cargo_names_after_it() {
        assert_eq!(
            [
                BuildFeature::Cuda,
                BuildFeature::CudaNoVmm,
                BuildFeature::DynamicBackends,
                BuildFeature::DynamicLink,
                BuildFeature::FaultInjection,
                BuildFeature::Metal,
                BuildFeature::Openmp,
                BuildFeature::Rocm,
                BuildFeature::SharedStdcxx,
                BuildFeature::StaticStdcxx,
                BuildFeature::SystemGgml,
                BuildFeature::SystemGgmlStatic,
                BuildFeature::Vulkan,
            ]
            .map(BuildFeature::cargo_variable),
            [
                "CARGO_FEATURE_CUDA",
                "CARGO_FEATURE_CUDA_NO_VMM",
                "CARGO_FEATURE_DYNAMIC_BACKENDS",
                "CARGO_FEATURE_DYNAMIC_LINK",
                "CARGO_FEATURE_FAULT_INJECTION",
                "CARGO_FEATURE_METAL",
                "CARGO_FEATURE_OPENMP",
                "CARGO_FEATURE_ROCM",
                "CARGO_FEATURE_SHARED_STDCXX",
                "CARGO_FEATURE_STATIC_STDCXX",
                "CARGO_FEATURE_SYSTEM_GGML",
                "CARGO_FEATURE_SYSTEM_GGML_STATIC",
                "CARGO_FEATURE_VULKAN",
            ]
        );
    }
}
