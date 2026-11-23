use crate::fitted_mtmd_input_chunk::FittedMtmdInputChunk;

/// A fitted chunk that llama.cpp has just evaluated, with the position after it.
#[derive(Debug)]
pub struct EvaluatedMtmdInputChunk<'fitted, 'chunks, 'context> {
    pub fitted_chunk: &'fitted FittedMtmdInputChunk<'chunks, 'context>,
    pub next_position: llama_cpp_bindings_sys::bindings::llama_pos,
}
