use crate::model_error::ModelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionType {
    None,
    OpenJev,
    Lev,
    Kev,
    Nimble,
    Laya,
    Clef,
    PplxDecider,
    VllmDecision2,
    VllmD3,
    Lfm2D1,
    Lfm2D1Omni,
    GraniteSwitch,
    H2oLightning,
    MinicpmOcrJev,
    Lfm3D1,
    Lfm3D1Omni,
    Unknown,
}

impl TryFrom<llama_cpp_bindings_sys::bindings::llama_rs_decision_type> for DecisionType {
    type Error = ModelError;

    fn try_from(
        value: llama_cpp_bindings_sys::bindings::llama_rs_decision_type,
    ) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_OPENJEV => Ok(Self::OpenJev),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LEV => Ok(Self::Lev),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_KEV => Ok(Self::Kev),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_NIMBLE => Ok(Self::Nimble),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LAYA => Ok(Self::Laya),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_CLEF => Ok(Self::Clef),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_PPLX_DECIDER => {
                Ok(Self::PplxDecider)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_VLLM_DECISION2 => {
                Ok(Self::VllmDecision2)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_VLLM_D3 => Ok(Self::VllmD3),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM2_D1 => Ok(Self::Lfm2D1),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM2_D1_OMNI => {
                Ok(Self::Lfm2D1Omni)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_GRANITE_SWITCH => {
                Ok(Self::GraniteSwitch)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_H2O_LIGHTNING => {
                Ok(Self::H2oLightning)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_MINICPM_OCR_JEV => {
                Ok(Self::MinicpmOcrJev)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM3_D1 => Ok(Self::Lfm3D1),
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM3_D1_OMNI => {
                Ok(Self::Lfm3D1Omni)
            }
            llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_UNKNOWN => Ok(Self::Unknown),
            unknown => Err(ModelError::UnknownDecisionType(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decision_type::DecisionType;
    use crate::model_error::ModelError;

    #[test]
    fn every_defined_decision_type_converts_to_its_variant() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_NONE,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_OPENJEV,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LEV,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_KEV,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_NIMBLE,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LAYA,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_CLEF,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_PPLX_DECIDER,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_VLLM_DECISION2,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_VLLM_D3,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM2_D1,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM2_D1_OMNI,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_GRANITE_SWITCH,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_H2O_LIGHTNING,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_MINICPM_OCR_JEV,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM3_D1,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_LFM3_D1_OMNI,
                llama_cpp_bindings_sys::bindings::LLAMA_RS_DECISION_TYPE_UNKNOWN,
            ]
            .map(DecisionType::try_from),
            [
                Ok(DecisionType::None),
                Ok(DecisionType::OpenJev),
                Ok(DecisionType::Lev),
                Ok(DecisionType::Kev),
                Ok(DecisionType::Nimble),
                Ok(DecisionType::Laya),
                Ok(DecisionType::Clef),
                Ok(DecisionType::PplxDecider),
                Ok(DecisionType::VllmDecision2),
                Ok(DecisionType::VllmD3),
                Ok(DecisionType::Lfm2D1),
                Ok(DecisionType::Lfm2D1Omni),
                Ok(DecisionType::GraniteSwitch),
                Ok(DecisionType::H2oLightning),
                Ok(DecisionType::MinicpmOcrJev),
                Ok(DecisionType::Lfm3D1),
                Ok(DecisionType::Lfm3D1Omni),
                Ok(DecisionType::Unknown),
            ]
        );
    }

    #[test]
    fn a_value_llama_rs_decision_type_does_not_define_is_refused() {
        assert_eq!(
            DecisionType::try_from(9999),
            Err(ModelError::UnknownDecisionType(9999))
        );
    }
}
