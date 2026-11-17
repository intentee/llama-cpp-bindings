#[derive(Clone, Debug, PartialEq)]
pub struct GgufTensorF32 {
    pub shape: [i64; llama_cpp_bindings_sys::GGML_MAX_DIMS as usize],
    pub values: Vec<f32>,
}
