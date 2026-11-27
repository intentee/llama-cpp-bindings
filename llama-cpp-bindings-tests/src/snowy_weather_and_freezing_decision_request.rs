use indexmap::IndexMap;
use llama_cpp_bindings_types::decision_question::DecisionQuestion;
use llama_cpp_bindings_types::decision_request::DecisionRequest;
use serde_json::json;

use crate::snowy_weather_decision_request::snowy_weather_decision_request;

#[must_use]
pub fn snowy_weather_and_freezing_decision_request() -> DecisionRequest {
    let mut request = snowy_weather_decision_request();

    request.questions.insert(
        "freezing".to_owned(),
        DecisionQuestion::Noul {
            criteria: IndexMap::new(),
            instructions: json!("Is it below freezing outside?"),
        },
    );

    request
}
