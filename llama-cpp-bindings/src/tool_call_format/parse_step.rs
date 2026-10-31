use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;

#[derive(Debug)]
pub enum ParseStep<'body> {
    Done,
    Call {
        call: ParsedToolCall,
        remainder: &'body str,
    },
}
