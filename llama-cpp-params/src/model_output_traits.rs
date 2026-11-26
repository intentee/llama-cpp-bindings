use llama_cpp_bindings_sys::bindings::LLAMA_FLASH_ATTN_TYPE_DISABLED;
use llama_cpp_bindings_sys::bindings::llama_rs_fit_model_traits;

use crate::llama_context_params::LlamaContextParams;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelOutputTraits {
    pub causal_attention: bool,
    pub has_encoder: bool,
    pub n_ctx_train: u32,
}

impl ModelOutputTraits {
    #[must_use]
    pub const fn context_params(self, context_params: LlamaContextParams) -> LlamaContextParams {
        if self.has_encoder && context_params.n_seq_max() > 1 {
            context_params
                .with_kv_unified(true)
                .with_flash_attention_policy(LLAMA_FLASH_ATTN_TYPE_DISABLED)
        } else {
            context_params
        }
    }
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
