use crate::cmake_define::CmakeDefine;

/// A ggml x86 instruction-set switch and the rustc target features that must all be enabled
/// to turn it on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct X86GgmlSwitch {
    pub name: &'static str,
    pub target_features: &'static [&'static str],
}

impl X86GgmlSwitch {
    #[must_use]
    pub fn define_for(&self, enabled_features: &[&str]) -> CmakeDefine {
        CmakeDefine::switch(
            self.name,
            self.target_features
                .iter()
                .all(|target_feature| enabled_features.contains(target_feature)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::X86GgmlSwitch;
    use crate::cmake_define::CmakeDefine;

    const AVX512: X86GgmlSwitch = X86GgmlSwitch {
        name: "GGML_AVX512",
        target_features: &["avx512f", "avx512bw"],
    };

    #[test]
    fn a_switch_is_on_only_when_every_target_feature_it_needs_is_enabled() {
        assert_eq!(
            [
                AVX512.define_for(&["avx512f", "avx512bw", "avx2"]),
                AVX512.define_for(&["avx512f", "avx2"]),
            ],
            [
                CmakeDefine::switch("GGML_AVX512", true),
                CmakeDefine::switch("GGML_AVX512", false),
            ]
        );
    }
}
