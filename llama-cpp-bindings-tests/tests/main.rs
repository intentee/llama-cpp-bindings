#![expect(
    clippy::unnecessary_wraps,
    reason = "harness trial functions must return Result to satisfy LlamaTestFn registration"
)]
mod backend_initialization;
mod chat_protocol;
mod context_state;
mod embedding_models;
mod embedding_row_width;
mod generation_control;
mod hidden_states;
mod model_derived_data_caching;
mod model_introspection;
mod model_loading_errors;
mod multimodal_audio;
mod multimodal_fusion;
mod multimodal_prompt_parts_tokenization;
mod multimodal_vision;
mod sequence_embedding_availability;
mod structured_chat_output;

llama_cpp_test_harness::llama_tests_main!();
