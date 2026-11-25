use llama_cpp_bindings_sys::bindings::llama_rs_decision_order;

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
    pub const fn from_raw(raw: llama_rs_decision_order) -> Self {
        match raw {
            llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_NONE => Self::None,
            llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_NOUL => Self::QuestionNoul,
            llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE => {
                Self::QuestionChoice
            }
            llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_SCORE => Self::QuestionScore,
            llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_OPTION => Self::Option,
        }
    }

    #[must_use]
    pub const fn to_raw(self) -> llama_rs_decision_order {
        match self {
            Self::None => llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_NONE,
            Self::QuestionNoul => llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_NOUL,
            Self::QuestionChoice => {
                llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE
            }
            Self::QuestionScore => llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_SCORE,
            Self::Option => llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_OPTION,
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_sys::bindings::llama_rs_decision_order;

    use crate::decision_order::DecisionOrder;

    #[test]
    fn every_llama_cpp_decision_order_converts_back_to_its_decision_order() {
        let decision_orders = [
            DecisionOrder::None,
            DecisionOrder::QuestionNoul,
            DecisionOrder::QuestionChoice,
            DecisionOrder::QuestionScore,
            DecisionOrder::Option,
        ];

        assert_eq!(
            decision_orders.map(|decision_order| DecisionOrder::from_raw(decision_order.to_raw())),
            decision_orders
        );
    }

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
                llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_NONE,
                llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_NOUL,
                llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE,
                llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_QUESTION_SCORE,
                llama_rs_decision_order::LLAMA_RS_DECISION_ORDER_OPTION,
            ]
        );
    }
}
