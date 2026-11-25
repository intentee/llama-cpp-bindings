use indexmap::IndexMap;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use crate::decision_question::DecisionQuestion;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    pub questions: IndexMap<String, DecisionQuestion>,
    pub state: Value,
}

impl From<&DecisionRequest> for Value {
    fn from(
        DecisionRequest {
            images,
            questions,
            state,
        }: &DecisionRequest,
    ) -> Self {
        let mut request = Map::new();

        if !images.is_empty() {
            request.insert(
                "images".to_owned(),
                Self::Array(images.iter().cloned().map(Self::String).collect()),
            );
        }
        request.insert(
            "questions".to_owned(),
            Self::Object(
                questions
                    .iter()
                    .map(|(question_id, question)| (question_id.clone(), Self::from(question)))
                    .collect(),
            ),
        );
        request.insert("state".to_owned(), state.clone());

        Self::Object(request)
    }
}
