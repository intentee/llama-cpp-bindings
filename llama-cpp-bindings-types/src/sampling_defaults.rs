use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    pub min_p: f32,
    pub penalty_frequency: f32,
    pub penalty_last_n: i32,
    pub penalty_presence: f32,
    pub penalty_repeat: f32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
}
