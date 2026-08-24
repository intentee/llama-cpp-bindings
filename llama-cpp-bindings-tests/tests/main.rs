mod backend_initialization;
mod chat_protocol;
mod context_state;
mod embedding_models;
mod generation_control;
mod model_introspection;
mod model_loading_errors;
mod multimodal_audio;
mod multimodal_fusion;
mod multimodal_vision;
mod structured_chat_output;

llama_cpp_test_harness::llama_tests_main!();
