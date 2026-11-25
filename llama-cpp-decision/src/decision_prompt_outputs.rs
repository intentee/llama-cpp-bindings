use llama_cpp_bindings_sys::bindings::llama_seq_id;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionPromptOutputs<'batch> {
    pub batch_indices: &'batch [i32],
    pub sequence_id: llama_seq_id,
}
