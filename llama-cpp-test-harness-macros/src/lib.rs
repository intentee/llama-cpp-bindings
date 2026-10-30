#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod expand;
mod parsed_args;
mod parsed_context_params;
mod parsed_load_mode;
mod parsed_model_load_params;
mod parsed_source;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

use crate::expand::expand;

fn dispatch(attribute: TokenStream2, item: TokenStream2) -> TokenStream2 {
    match expand(attribute, item) {
        Ok(tokens) => tokens,
        Err(error) => error.to_compile_error(),
    }
}

#[proc_macro_attribute]
pub fn llama_test(attribute: TokenStream, item: TokenStream) -> TokenStream {
    dispatch(attribute.into(), item.into()).into()
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::dispatch;
    use crate::expand::expand;

    #[test]
    fn dispatch_on_invalid_attribute_emits_compile_error_tokens() {
        let attribute = quote! { totally_wrong = "x" };
        let item = quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let message = "unknown field `totally_wrong`; expected one of: model_source, \
                       n_gpu_layers, load_mode, n_ctx, n_batch, n_ubatch, mmproj_source, \
                       embeddings, n_seq_max, n_threads_batch, void_logs";

        assert_eq!(
            dispatch(attribute, item).to_string(),
            quote! { ::core::compile_error! { #message } }.to_string()
        );
    }

    #[test]
    fn dispatch_on_valid_input_emits_inventory_submission() {
        let attribute = quote! {
            model_source = HuggingFace("r", "f"),
            n_gpu_layers = 0,
            load_mode = Mmap,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1
        };
        let item = quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        };
        let expanded = expand(attribute.clone(), item.clone()).expect("valid input must expand");

        assert_eq!(dispatch(attribute, item).to_string(), expanded.to_string());
    }
}
