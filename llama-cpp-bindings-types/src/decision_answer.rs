use indexmap::IndexMap;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase", tag = "type")]
pub enum DecisionAnswer {
    Choice {
        choice: String,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
    },
    Noul {
        noul: f64,
    },
    Score {
        confidence: f64,
        legend: IndexMap<String, Value>,
        probabilities: IndexMap<String, f64>,
        score: f64,
    },
}
