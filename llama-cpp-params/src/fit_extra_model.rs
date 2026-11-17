use std::ffi::CStr;

use crate::llama_context_params::LlamaContextParams;

/// A second model that shares the devices of the model being fitted, such as a draft model.
///
/// Its `model_params` are the `params` of its own `LlamaModelParams`. Its context follows the
/// fitted model's context, so `context_params` is rewritten by `LlamaModelParams::fit_params`.
/// Set `shares_model` when the weights are already
/// accounted for by the fitted model, as they are for an MTP context.
pub struct FitExtraModel<'extra_model> {
    pub model_path: &'extra_model CStr,
    pub model_params: &'extra_model mut llama_cpp_bindings_sys::llama_model_params,
    pub context_params: &'extra_model mut LlamaContextParams,
    pub shares_model: bool,
}

impl FitExtraModel<'_> {
    pub const fn as_ffi(&mut self) -> llama_cpp_bindings_sys::llama_rs_fit_extra_model {
        llama_cpp_bindings_sys::llama_rs_fit_extra_model {
            path_model: self.model_path.as_ptr(),
            mparams: &raw mut *self.model_params,
            cparams: &raw mut self.context_params.context_params,
            shares_model: self.shares_model,
        }
    }
}
