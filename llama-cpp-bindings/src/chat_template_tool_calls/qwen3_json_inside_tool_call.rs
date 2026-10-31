use llama_cpp_bindings_types::json_object_shape::JsonObjectShape;
use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;
use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;

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
