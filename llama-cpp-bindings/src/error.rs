pub mod chat_session_error;
pub mod eval_multimodal_chunks_error;
pub mod ingest_error;
pub mod llama_cpp_error;
pub mod parse_chat_message_error;

use llama_cpp_error::LlamaCppError;

pub type Result<TValue> = std::result::Result<TValue, LlamaCppError>;
