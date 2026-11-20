use crate::llama_lora_adapter::LlamaLoraAdapter;

#[derive(Debug)]
pub struct LoraAdapterScale<'model> {
    pub adapter: LlamaLoraAdapter<'model>,
    pub scale: f32,
}
