use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::token::LlamaToken;

/// # Panics
/// Panics when `text` cannot be tokenized or does not tokenize into exactly one token.
#[must_use]
pub fn single_token(model: &LlamaModel, text: &str) -> LlamaToken {
    let [token]: [LlamaToken; 1] = model
        .str_to_token(text, AddBos::Never, ParseSpecialTokens::Always)
        .expect("the text must tokenize")
        .try_into()
        .expect("the text must tokenize into exactly one token");

    token
}
