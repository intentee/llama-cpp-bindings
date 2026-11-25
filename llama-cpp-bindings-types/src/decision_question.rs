use indexmap::IndexMap;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase", tag = "type")]
pub enum DecisionQuestion {
    Choice {
        criteria: IndexMap<String, Value>,
        #[serde(default, skip_serializing_if = "Value::is_null")]
        instructions: Value,
    },
    Noul {
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        criteria: IndexMap<String, Value>,
        #[serde(default, skip_serializing_if = "Value::is_null")]
        instructions: Value,
    },
    Score {
        criteria: Vec<Value>,
        #[serde(default, skip_serializing_if = "Value::is_null")]
        instructions: Value,
    },
}

fn criteria_object(criteria: &IndexMap<String, Value>) -> Value {
    Value::Object(
        criteria
            .iter()
            .map(|(key, description)| (key.clone(), description.clone()))
            .collect(),
    )
}

fn question_object(kind: &str, criteria: Value, instructions: &Value) -> Value {
    let mut question = Map::new();

    question.insert("type".to_owned(), Value::String(kind.to_owned()));
    question.insert("criteria".to_owned(), criteria);

    if !instructions.is_null() {
        question.insert("instructions".to_owned(), instructions.clone());
    }

    Value::Object(question)
}

impl From<&DecisionQuestion> for Value {
    fn from(question: &DecisionQuestion) -> Self {
        match question {
            DecisionQuestion::Choice {
                criteria,
                instructions,
            } => question_object("choice", criteria_object(criteria), instructions),
            DecisionQuestion::Noul {
                criteria,
                instructions,
            } => question_object("noul", criteria_object(criteria), instructions),
            DecisionQuestion::Score {
                criteria,
                instructions,
            } => question_object("score", Self::Array(criteria.clone()), instructions),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::DecisionQuestion;

    #[test]
    fn a_score_question_carries_its_levels_to_llama_cpp_in_order() {
        assert_eq!(
            Value::from(&DecisionQuestion::Score {
                criteria: vec![json!("cold"), json!("warm"), json!("hot")],
                instructions: json!("How warm is it?"),
            }),
            json!({
                "type": "score",
                "criteria": ["cold", "warm", "hot"],
                "instructions": "How warm is it?",
            })
        );
    }
}
