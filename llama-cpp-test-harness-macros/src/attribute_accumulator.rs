use crate::parsed_load_mode::ParsedLoadMode;
use crate::parsed_source::ParsedSource;

#[derive(Default)]
pub struct AttributeAccumulator {
    pub model_source: Option<ParsedSource>,
    pub mmproj_source: Option<ParsedSource>,
    pub n_gpu_layers: Option<i32>,
    pub load_mode: Option<ParsedLoadMode>,
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub embeddings: Option<bool>,
    pub n_seq_max: Option<u32>,
    pub n_threads_batch: Option<i32>,
    pub void_logs: Option<bool>,
}
