use serde::Deserialize;
use serde::Serialize;

use crate::sampling_defaults::SamplingDefaults;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefaults {
    pub context_size: u32,
    pub sampling: SamplingDefaults,
}
