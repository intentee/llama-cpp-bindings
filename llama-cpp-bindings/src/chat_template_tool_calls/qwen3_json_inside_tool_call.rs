use llama_cpp_bindings_types::JsonObjectShape;
use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;

pub struct Qwen3JsonInsideToolCallFormat;

impl Qwen3JsonInsideToolCallFormat {
    #[must_use]
    pub fn markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<tool_call>".to_owned(),
            close: "</tool_call>".to_owned(),
            args_shape: ToolCallArgsShape::JsonObject(JsonObjectShape {
                name_field: "name".to_owned(),
                arguments_field: "arguments".to_owned(),
            }),
        }
    }
}
