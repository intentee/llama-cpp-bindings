use llama_cpp_bindings_types::ParsedToolCall;

#[derive(Debug)]
pub enum ParseStep<'body> {
    Done,
    Call {
        call: ParsedToolCall,
        remainder: &'body str,
    },
}
