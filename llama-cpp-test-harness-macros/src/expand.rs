use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Item;
use syn::ItemFn;
use syn::ReturnType;
use syn::Signature;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::parse2;

use crate::parsed_args::ParsedArgs;
use crate::parsed_source::ParsedSource;

struct StackedItems {
    items: Vec<Item>,
}

impl Parse for StackedItems {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut items = Vec::new();
        while !input.is_empty() {
            items.push(input.parse()?);
        }
        Ok(Self { items })
    }
}

fn validate_signature(item_fn: &ItemFn) -> syn::Result<()> {
    if item_fn.sig.inputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.inputs,
            "llama_test functions must take exactly one argument: `&LlamaFixture<'_>`",
        ));
    }
    if item_fn.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "llama_test functions must be synchronous",
        ));
    }
    if !item_fn.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.generics,
            "llama_test functions must not be generic",
        ));
    }
    Ok(())
}

fn split_fn_and_pass_through(items: Vec<Item>) -> syn::Result<(ItemFn, Vec<Item>)> {
    let mut found_fn: Option<ItemFn> = None;
    let mut pass_through: Vec<Item> = Vec::new();
    for parsed_item in items {
        match parsed_item {
            Item::Fn(item_fn) => {
                if found_fn.is_some() {
                    return Err(syn::Error::new_spanned(
                        &item_fn,
                        "llama_test expects exactly one fn definition",
                    ));
                }
                found_fn = Some(item_fn);
            }
            other => pass_through.push(other),
        }
    }
    let item_fn = found_fn
        .ok_or_else(|| syn::Error::new(Span::call_site(), "llama_test expects an fn definition"))?;
    Ok((item_fn, pass_through))
}

fn build_model_source_literal(source: &ParsedSource) -> TokenStream {
    match source {
        ParsedSource::HuggingFace { repo, file } => quote! {
            ::llama_cpp_test_harness::GgufSource::HuggingFace {
                repo: #repo,
                file: #file,
            }
        },
        ParsedSource::LocalPath(path) => quote! {
            ::llama_cpp_test_harness::GgufSource::LocalPath(#path)
        },
    }
}

fn build_mmproj_source_literal(source: Option<&ParsedSource>) -> TokenStream {
    match source {
        None => quote! { ::core::option::Option::None },
        Some(ParsedSource::HuggingFace { repo, file }) => quote! {
            ::core::option::Option::Some(::llama_cpp_test_harness::GgufSource::HuggingFace {
                repo: #repo,
                file: #file,
            })
        },
        Some(ParsedSource::LocalPath(path)) => quote! {
            ::core::option::Option::Some(::llama_cpp_test_harness::GgufSource::LocalPath(#path))
        },
    }
}

fn build_test_fn_literal(signature: &Signature) -> TokenStream {
    let fn_name = &signature.ident;

    match signature.output {
        ReturnType::Default => {
            quote! { ::llama_cpp_test_harness::LlamaTestFn::Infallible(#fn_name) }
        }
        ReturnType::Type(..) => {
            quote! { ::llama_cpp_test_harness::LlamaTestFn::Fallible(#fn_name) }
        }
    }
}

fn build_registration(args: &ParsedArgs, signature: &Signature) -> TokenStream {
    let fn_name = &signature.ident;
    let trial_name = format!(
        "{fn_name}[{suffix}]",
        suffix = args.model_source.display_suffix()
    );
    let model_source_literal = build_model_source_literal(&args.model_source);
    let mmproj_source_literal = build_mmproj_source_literal(args.mmproj_source.as_ref());
    let gpu_layers = args.model_load_params.n_gpu_layers;
    let load_mode = args.model_load_params.load_mode.tokens();
    let context_size = args.context_params.n_ctx;
    let logical_batch = args.context_params.n_batch;
    let physical_batch = args.context_params.n_ubatch;
    let embeddings_flag = args.context_params.embeddings;
    let sequence_max = args.context_params.n_seq_max;
    let void_logs_flag = args.void_logs;
    let test_fn = build_test_fn_literal(signature);
    let threads_batch = args.context_params.n_threads_batch.map_or_else(
        || quote! { ::core::option::Option::None },
        |value| quote! { ::core::option::Option::Some(#value) },
    );

    quote! {
        ::llama_cpp_test_harness::inventory::submit! {
            ::llama_cpp_test_harness::LlamaTestRegistration {
                name: #trial_name,
                key: ::llama_cpp_test_harness::LoadKey {
                    model_source: #model_source_literal,
                    mmproj_source: #mmproj_source_literal,
                    model_load_params: ::llama_cpp_test_harness::ModelLoadParams {
                        n_gpu_layers: #gpu_layers,
                        load_mode: #load_mode,
                    },
                },
                context_params: ::llama_cpp_test_harness::ContextParams {
                    n_ctx: #context_size,
                    n_batch: #logical_batch,
                    n_ubatch: #physical_batch,
                    n_seq_max: #sequence_max,
                    n_threads_batch: #threads_batch,
                    embeddings: #embeddings_flag,
                },
                void_logs: #void_logs_flag,
                func: #test_fn,
            }
        }
    }
}

pub fn expand(attribute: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: ParsedArgs = parse2(attribute)?;
    let StackedItems { items } = parse2(item)?;
    let (item_fn, pass_through) = split_fn_and_pass_through(items)?;
    validate_signature(&item_fn)?;

    let new_submission = build_registration(&args, &item_fn.sig);

    Ok(quote! {
        #item_fn
        #(#pass_through)*
        #new_submission
    })
}

#[cfg(test)]
mod tests {
    use proc_macro2::TokenStream;
    use quote::quote;
    use syn::Item;
    use syn::ItemFn;
    use syn::parse2;

    use super::StackedItems;
    use super::build_mmproj_source_literal;
    use super::build_model_source_literal;
    use super::build_test_fn_literal;
    use super::expand;
    use crate::parsed_source::ParsedSource;

    fn well_formed_attribute() -> TokenStream {
        quote! {
            model_source = HuggingFace("foo", "bar.gguf"),
            n_gpu_layers = 0,
            load_mode = Mmap,
            n_ctx = 1,
            n_batch = 1,
            n_ubatch = 1
        }
    }

    fn well_formed_function() -> TokenStream {
        quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
        }
    }

    fn expansion_error(attribute: TokenStream, item: TokenStream) -> String {
        expand(attribute, item)
            .expect_err("the input must be refused")
            .to_string()
    }

    fn test_fn_literal_of(item: TokenStream) -> String {
        let item_fn: ItemFn = parse2(item).expect("the item must be a function");

        build_test_fn_literal(&item_fn.sig).to_string()
    }

    #[test]
    fn expansion_keeps_the_function_and_submits_its_registration() {
        let expanded = expand(well_formed_attribute(), well_formed_function())
            .expect("well-formed input must expand");
        let expected = quote! {
            fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
            ::llama_cpp_test_harness::inventory::submit! {
                ::llama_cpp_test_harness::LlamaTestRegistration {
                    name: "my_test[bar.gguf]",
                    key: ::llama_cpp_test_harness::LoadKey {
                        model_source: ::llama_cpp_test_harness::GgufSource::HuggingFace {
                            repo: "foo",
                            file: "bar.gguf",
                        },
                        mmproj_source: ::core::option::Option::None,
                        model_load_params: ::llama_cpp_test_harness::ModelLoadParams {
                            n_gpu_layers: 0i32,
                            load_mode: ::llama_cpp_test_harness::LlamaLoadMode::Mmap,
                        },
                    },
                    context_params: ::llama_cpp_test_harness::ContextParams {
                        n_ctx: 1u32,
                        n_batch: 1u32,
                        n_ubatch: 1u32,
                        n_seq_max: 1u32,
                        n_threads_batch: ::core::option::Option::None,
                        embeddings: false,
                    },
                    void_logs: false,
                    func: ::llama_cpp_test_harness::LlamaTestFn::Fallible(my_test),
                }
            }
        };

        assert_eq!(expanded.to_string(), expected.to_string());
    }

    #[test]
    fn a_function_returning_a_result_registers_as_fallible() {
        assert_eq!(
            test_fn_literal_of(well_formed_function()),
            quote! { ::llama_cpp_test_harness::LlamaTestFn::Fallible(my_test) }.to_string()
        );
    }

    #[test]
    fn a_function_without_a_return_type_registers_as_infallible() {
        assert_eq!(
            test_fn_literal_of(quote! { fn my_test(fixture: &LlamaFixture<'_>) {} }),
            quote! { ::llama_cpp_test_harness::LlamaTestFn::Infallible(my_test) }.to_string()
        );
    }

    #[test]
    fn a_local_path_model_source_emits_the_local_path_variant() {
        assert_eq!(
            build_model_source_literal(&ParsedSource::LocalPath("/abs/local.gguf".to_owned()))
                .to_string(),
            quote! { ::llama_cpp_test_harness::GgufSource::LocalPath("/abs/local.gguf") }
                .to_string()
        );
    }

    #[test]
    fn a_huggingface_mmproj_source_is_wrapped_in_some() {
        assert_eq!(
            build_mmproj_source_literal(Some(&ParsedSource::HuggingFace {
                repo: "r".to_owned(),
                file: "mmproj.gguf".to_owned(),
            }))
            .to_string(),
            quote! {
                ::core::option::Option::Some(::llama_cpp_test_harness::GgufSource::HuggingFace {
                    repo: "r",
                    file: "mmproj.gguf",
                })
            }
            .to_string()
        );
    }

    #[test]
    fn a_local_path_mmproj_source_is_wrapped_in_some() {
        assert_eq!(
            build_mmproj_source_literal(Some(&ParsedSource::LocalPath(
                "/abs/mmproj.gguf".to_owned()
            )))
            .to_string(),
            quote! {
                ::core::option::Option::Some(::llama_cpp_test_harness::GgufSource::LocalPath(
                    "/abs/mmproj.gguf"
                ))
            }
            .to_string()
        );
    }

    #[test]
    fn a_malformed_attribute_propagates_the_parser_error() {
        assert_eq!(
            expansion_error(quote! { totally_wrong = "x" }, well_formed_function()),
            "unknown field `totally_wrong`; expected one of: model_source, n_gpu_layers, load_mode, \
             n_ctx, n_batch, n_ubatch, mmproj_source, embeddings, n_seq_max, n_threads_batch, \
             void_logs"
        );
    }

    #[test]
    fn a_function_with_two_arguments_is_refused() {
        assert_eq!(
            expansion_error(
                well_formed_attribute(),
                quote! { fn my_test(a: &LlamaFixture<'_>, b: i32) -> anyhow::Result<()> { Ok(()) } },
            ),
            "llama_test functions must take exactly one argument: `&LlamaFixture<'_>`"
        );
    }

    #[test]
    fn a_function_without_arguments_is_refused() {
        assert_eq!(
            expansion_error(
                well_formed_attribute(),
                quote! { fn my_test() -> anyhow::Result<()> { Ok(()) } },
            ),
            "llama_test functions must take exactly one argument: `&LlamaFixture<'_>`"
        );
    }

    #[test]
    fn an_async_function_is_refused() {
        assert_eq!(
            expansion_error(
                well_formed_attribute(),
                quote! {
                    async fn my_test(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
                },
            ),
            "llama_test functions must be synchronous"
        );
    }

    #[test]
    fn a_generic_function_is_refused() {
        assert_eq!(
            expansion_error(
                well_formed_attribute(),
                quote! {
                    fn my_test<T>(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
                },
            ),
            "llama_test functions must not be generic"
        );
    }

    #[test]
    fn an_impl_block_is_refused() {
        assert_eq!(
            expansion_error(
                well_formed_attribute(),
                quote! {
                    impl S {
                        fn my_test(&self) -> anyhow::Result<()> { Ok(()) }
                    }
                },
            ),
            "llama_test expects an fn definition"
        );
    }

    #[test]
    fn tokens_that_are_not_an_item_are_refused() {
        assert_eq!(
            expansion_error(well_formed_attribute(), quote! { this is not a function }),
            "expected `!`"
        );
    }

    #[test]
    fn two_function_definitions_are_refused() {
        assert_eq!(
            expansion_error(
                well_formed_attribute(),
                quote! {
                    fn first(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
                    fn second(fixture: &LlamaFixture<'_>) -> anyhow::Result<()> { Ok(()) }
                },
            ),
            "llama_test expects exactly one fn definition"
        );
    }

    #[test]
    fn a_stacked_invocation_keeps_the_registration_of_the_previous_layer() {
        let first_layer = expand(well_formed_attribute(), well_formed_function())
            .expect("the first layer must expand");
        let second_layer = expand(
            quote! {
                model_source = HuggingFace("second", "second.gguf"),
                n_gpu_layers = 1,
                load_mode = None,
                n_ctx = 2,
                n_batch = 2,
                n_ubatch = 2
            },
            first_layer,
        )
        .expect("the stacked layer must expand");
        let StackedItems { items } = parse2(second_layer).expect("the expansion must be items");

        assert_eq!(
            items
                .iter()
                .map(|item| matches!(item, Item::Fn(_)))
                .collect::<Vec<bool>>(),
            [true, false, false]
        );
        assert_eq!(
            items
                .iter()
                .map(|item| matches!(item, Item::Macro(_)))
                .collect::<Vec<bool>>(),
            [false, true, true]
        );
    }
}
