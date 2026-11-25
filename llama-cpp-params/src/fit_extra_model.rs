use std::path::Path;

use crate::llama_context_params::LlamaContextParams;

/// A second model that shares the devices of the model being fitted, such as a draft model.
///
/// `LlamaModelParams::fit_params` takes it with the extra model's own `LlamaModelParams` as
/// `model_params`. Its context follows the fitted model's context, so `context_params` is
/// rewritten by `LlamaModelParams::fit_params`. Set `shares_model` when the weights are already
/// accounted for by the fitted model, as they are for an MTP context.
pub struct FitExtraModel<'extra_model, TModelParams> {
    pub model_path: &'extra_model Path,
    pub model_params: &'extra_model mut TModelParams,
    pub context_params: &'extra_model mut LlamaContextParams,
    pub shares_model: bool,
}
