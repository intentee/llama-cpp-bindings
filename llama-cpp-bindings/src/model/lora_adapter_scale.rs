use crate::model::llama_lora_adapter::LlamaLoraAdapter;

#[derive(Debug)]
pub struct LoraAdapterScale<'adapter, 'model> {
    pub adapter: &'adapter LlamaLoraAdapter<'model>,
    pub scale: f32,
}
