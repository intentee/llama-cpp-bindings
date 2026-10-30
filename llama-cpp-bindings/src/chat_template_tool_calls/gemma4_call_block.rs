use llama_cpp_bindings_types::PairedQuoteShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;
use llama_cpp_bindings_types::ToolCallValueQuote;

pub struct Gemma4CallBlockFormat;

impl Gemma4CallBlockFormat {
    #[must_use]
    pub fn markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<|tool_call>call:".to_owned(),
            close: "}".to_owned(),
            args_shape: ToolCallArgsShape::PairedQuote(PairedQuoteShape {
                name_args_separator: "{".to_owned(),
                value_quote: ToolCallValueQuote {
                    open: "<|\"|>".to_owned(),
                    close: "<|\"|>".to_owned(),
                },
            }),
        }
    }
}
