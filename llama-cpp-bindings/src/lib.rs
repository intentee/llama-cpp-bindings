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
pub mod llama_utility;
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
    NewLlamaChatMessageError, Result, SamplerAcceptError, StringToTokenError, TokenToStringError,
};

pub use llama_backend_device::{
    LlamaBackendDevice, LlamaBackendDeviceType, list_llama_ggml_backend_devices,
};

pub use llama_utility::{
    ggml_time_us, json_schema_to_grammar, llama_time_us, max_devices, mlock_supported,
    mmap_supported, status_is_ok, status_to_i32,
};

pub use log_options::{LogOptions, send_logs_to_tracing};
