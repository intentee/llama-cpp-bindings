use std::ffi::CStr;
use std::pin::Pin;

use crate::context::params::LlamaContextParams;
use crate::model::params::LlamaModelParams;

/// A second model that shares the devices of the model being fitted, such as a draft model.
///
/// Its context follows the fitted model's context, so `context_params` is rewritten by
/// [`LlamaModelParams::fit_params`]. Set `shares_model` when the weights are already
/// accounted for by the fitted model, as they are for an MTP context.
pub struct FitExtraModel<'extra_model> {
    pub model_path: &'extra_model CStr,
    pub model_params: Pin<&'extra_model mut LlamaModelParams>,
    pub context_params: &'extra_model mut LlamaContextParams,
    pub shares_model: bool,
}

impl FitExtraModel<'_> {
    pub fn as_ffi(&mut self) -> llama_cpp_bindings_sys::llama_rs_fit_extra_model {
        llama_cpp_bindings_sys::llama_rs_fit_extra_model {
            path_model: self.model_path.as_ptr(),
            mparams: &raw mut self.model_params.params,
            cparams: &raw mut self.context_params.context_params,
            shares_model: self.shares_model,
        }
    }
}
