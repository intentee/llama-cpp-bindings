//! Bindings to the llama.cpp library.
//!
//! As llama.cpp is a very fast moving target, this crate does not attempt to create a stable API
//! with all the rust idioms. Instead it provided safe wrappers around nearly direct bindings to
//! llama.cpp. This makes it easier to keep up with the changes in llama.cpp, but does mean that
//! the API is not as nice as it could be.
//!
//! # Feature Flags
//!
//! - `cuda` enables CUDA gpu support.
//! - `sampler` adds the [`context::sample::sampler`] struct for a more rusty way of sampling.

pub mod context;
pub mod error;
pub mod llama_backend;
pub mod llama_backend_device;
pub mod llama_batch;
pub mod llama_utility_ggml_time_us;
pub mod llama_utility_json_schema_to_grammar;
pub mod llama_utility_llama_time_us;
pub mod llama_utility_max_devices;
pub mod llama_utility_mlock_supported;
pub mod llama_utility_mmap_supported;
pub mod llama_utility_status_is_ok;
pub mod llama_utility_status_to_i32;
#[cfg(feature = "llguidance")]
pub mod llguidance_sampler;
pub(crate) mod log;
pub mod log_options;
pub mod model;
#[cfg(feature = "mtmd")]
pub mod mtmd;
pub mod openai;
pub mod sampling;
pub mod timing;
pub mod token;
pub mod token_type;

pub use error::{
    ApplyChatTemplateError, ChatParseError, ChatTemplateError, DecodeError, EmbeddingsError,
    EncodeError, GrammarError, LlamaContextLoadError, LlamaCppError, LlamaLoraAdapterInitError,
    LlamaLoraAdapterRemoveError, LlamaLoraAdapterSetError, LlamaModelLoadError, MetaValError,
    ModelParamsError, NewLlamaChatMessageError, Result, SamplerAcceptError, SamplingError,
    StringToTokenError, TokenSamplingError, TokenToStringError,
};

pub use llama_backend_device::{
    LlamaBackendDevice, LlamaBackendDeviceType, list_llama_ggml_backend_devices,
};

pub use llama_utility_ggml_time_us::ggml_time_us;
pub use llama_utility_json_schema_to_grammar::json_schema_to_grammar;
pub use llama_utility_llama_time_us::llama_time_us;
pub use llama_utility_max_devices::max_devices;
pub use llama_utility_mlock_supported::mlock_supported;
pub use llama_utility_mmap_supported::mmap_supported;
pub use llama_utility_status_is_ok::status_is_ok;
pub use llama_utility_status_to_i32::status_to_i32;

pub use log::send_logs_to_tracing;
pub use log_options::LogOptions;
