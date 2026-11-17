use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::token::LlamaToken;

/// # Errors
/// Fails when `text` cannot be tokenized or does not tokenize into exactly one token.
pub fn single_token(model: &LlamaModel, text: &str) -> Result<LlamaToken> {
    let tokens = model
        .str_to_token(text, AddBos::Never, ParseSpecialTokens::Always)
        .context("the text must tokenize")?;
    let [token]: [LlamaToken; 1] = tokens
        .as_slice()
        .try_into()
        .context("the text must tokenize into exactly one token")?;

    Ok(token)
}
