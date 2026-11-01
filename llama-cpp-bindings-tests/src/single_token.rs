use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::token::LlamaToken;

/// # Errors
/// Forwards tokenization errors, and fails when `text` tokenizes into anything but one token.
pub fn single_token(model: &LlamaModel, text: &str) -> Result<LlamaToken> {
    let [token] = model.str_to_token(text, AddBos::Never, ParseSpecialTokens::Always)?[..] else {
        bail!("{text:?} must be a single token");
    };

    Ok(token)
}
