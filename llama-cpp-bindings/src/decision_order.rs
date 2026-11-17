#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionOrder {
    None,
    QuestionNoul,
    QuestionChoice,
    QuestionScore,
    Option,
}

impl DecisionOrder {
    #[must_use]
    pub const fn to_raw(self) -> llama_cpp_bindings_sys::llama_rs_decision_order {
        match self {
            Self::None => llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_NONE,
            Self::QuestionNoul => llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_QUESTION_NOUL,
            Self::QuestionChoice => llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE,
            Self::QuestionScore => llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_QUESTION_SCORE,
            Self::Option => llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_OPTION,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decision_order::DecisionOrder;

    #[test]
    fn every_decision_order_converts_to_its_llama_cpp_value() {
        assert_eq!(
            [
                DecisionOrder::None,
                DecisionOrder::QuestionNoul,
                DecisionOrder::QuestionChoice,
                DecisionOrder::QuestionScore,
                DecisionOrder::Option,
            ]
            .map(DecisionOrder::to_raw),
            [
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_NONE,
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_QUESTION_NOUL,
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE,
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_QUESTION_SCORE,
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_OPTION,
            ]
        );
    }
}
