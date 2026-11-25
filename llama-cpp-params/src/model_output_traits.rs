use llama_cpp_bindings_sys::bindings::llama_rs_fit_model_traits;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelOutputTraits {
    pub causal_attention: bool,
    pub has_encoder: bool,
    pub n_ctx_train: u32,
}

impl From<llama_rs_fit_model_traits> for ModelOutputTraits {
    fn from(
        llama_rs_fit_model_traits {
            causal_attention,
            has_encoder,
            n_ctx_train,
        }: llama_rs_fit_model_traits,
    ) -> Self {
        Self {
            causal_attention,
            has_encoder,
            n_ctx_train,
        }
    }
}
