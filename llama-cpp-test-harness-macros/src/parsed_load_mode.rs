use proc_macro2::Ident;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParsedLoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl ParsedLoadMode {
    pub fn parse(identifier: &Ident) -> syn::Result<Self> {
        match identifier.to_string().as_str() {
            "Auto" => Ok(Self::Auto),
            "None" => Ok(Self::None),
            "Mmap" => Ok(Self::Mmap),
            "Mlock" => Ok(Self::Mlock),
            "MmapMlock" => Ok(Self::MmapMlock),
            "DirectIo" => Ok(Self::DirectIo),
            _ => Err(syn::Error::new_spanned(
                identifier,
                "expected one of: Auto, None, Mmap, Mlock, MmapMlock, DirectIo",
            )),
        }
    }

    pub fn tokens(self) -> TokenStream {
        match self {
            Self::Auto => quote! { ::llama_cpp_test_harness::LlamaLoadMode::Auto },
            Self::None => quote! { ::llama_cpp_test_harness::LlamaLoadMode::None },
            Self::Mmap => quote! { ::llama_cpp_test_harness::LlamaLoadMode::Mmap },
            Self::Mlock => quote! { ::llama_cpp_test_harness::LlamaLoadMode::Mlock },
            Self::MmapMlock => quote! { ::llama_cpp_test_harness::LlamaLoadMode::MmapMlock },
            Self::DirectIo => quote! { ::llama_cpp_test_harness::LlamaLoadMode::DirectIo },
        }
    }
}

#[cfg(test)]
mod tests {
    use proc_macro2::Ident;
    use proc_macro2::Span;

    use super::ParsedLoadMode;

    #[test]
    fn every_load_mode_identifier_parses_and_emits_its_public_variant() {
        let cases = [
            ("Auto", ParsedLoadMode::Auto),
            ("None", ParsedLoadMode::None),
            ("Mmap", ParsedLoadMode::Mmap),
            ("Mlock", ParsedLoadMode::Mlock),
            ("MmapMlock", ParsedLoadMode::MmapMlock),
            ("DirectIo", ParsedLoadMode::DirectIo),
        ];

        for (identifier, expected) in cases {
            let identifier = Ident::new(identifier, Span::call_site());
            let parsed = ParsedLoadMode::parse(&identifier).expect("known mode must parse");

            assert_eq!(parsed, expected);
            assert_eq!(
                parsed.tokens().to_string(),
                format!(":: llama_cpp_test_harness :: LlamaLoadMode :: {identifier}"),
            );
        }
    }

    #[test]
    fn unknown_load_mode_identifier_reports_every_valid_choice() {
        let identifier = Ident::new("Buffered", Span::call_site());
        let message = ParsedLoadMode::parse(&identifier)
            .expect_err("unknown mode must fail")
            .to_string();

        for valid in ["Auto", "None", "Mmap", "Mlock", "MmapMlock", "DirectIo"] {
            assert!(message.contains(valid), "missing {valid} in {message}");
        }
    }
}
