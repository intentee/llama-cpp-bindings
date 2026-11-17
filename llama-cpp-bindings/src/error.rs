pub mod batch_transfer_error;
pub mod chat_session_error;
pub mod decode_error;
pub mod embeddings_error;
pub mod encode_error;
pub mod eval_multimodal_chunks_error;
pub mod grammar_error;
pub mod grammar_runtime_error;
pub mod ingest_error;
pub mod llama_context_load_error;
pub mod llama_cpp_error;
pub mod llama_lora_adapters_error;
pub mod logits_error;
pub mod memory_seq_add_error;
pub mod memory_seq_cp_error;
pub mod memory_seq_div_error;
pub mod memory_seq_pos_max_error;
pub mod memory_seq_rm_error;
pub mod parse_chat_message_error;
pub mod process_failure;
pub mod sample_error;
pub mod sampler_accept_error;
pub mod sampler_apply_error;
pub mod sampling_error;
pub mod sequence_id_out_of_range;

use llama_cpp_error::LlamaCppError;

pub type Result<TValue> = std::result::Result<TValue, LlamaCppError>;
