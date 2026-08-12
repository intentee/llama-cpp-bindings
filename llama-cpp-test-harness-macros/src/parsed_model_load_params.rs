use crate::parsed_load_mode::ParsedLoadMode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsedModelLoadParams {
    pub n_gpu_layers: i32,
    pub load_mode: ParsedLoadMode,
}
