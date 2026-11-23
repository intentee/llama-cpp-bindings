use enumflags2::bitflags;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[bitflags]
#[repr(u16)]
pub enum BuildFeature {
    Cuda = 1 << 0,
    CudaNoVmm = 1 << 1,
    DynamicBackends = 1 << 2,
    DynamicLink = 1 << 3,
    FaultInjection = 1 << 4,
    Metal = 1 << 5,
    Openmp = 1 << 6,
    Rocm = 1 << 7,
    SharedStdcxx = 1 << 8,
    StaticStdcxx = 1 << 9,
    SystemGgml = 1 << 10,
    SystemGgmlStatic = 1 << 11,
    Vulkan = 1 << 12,
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
    use enumflags2::BitFlags;

    use super::BuildFeature;

    #[test]
    fn every_feature_is_read_from_the_variable_cargo_names_after_it() {
        assert_eq!(
            BitFlags::<BuildFeature>::all()
                .iter()
                .map(BuildFeature::cargo_variable)
                .collect::<Vec<&str>>(),
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
