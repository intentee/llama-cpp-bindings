use llama_cpp_bindings_types::BracketedJsonShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;

pub struct Mistral3ArrowArgsFormat;

impl Mistral3ArrowArgsFormat {
    #[must_use]
    pub fn markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "[TOOL_CALLS]".to_owned(),
            close: String::new(),
            args_shape: ToolCallArgsShape::BracketedJson(BracketedJsonShape {
                name_args_separator: "[ARGS]".to_owned(),
            }),
        }
    }
}
