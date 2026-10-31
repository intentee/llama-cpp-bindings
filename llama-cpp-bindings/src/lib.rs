#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod backend_device_index;
pub mod bare_json_tool_call_probe;
pub mod bare_json_tool_calls;
pub mod batch_add_error;
pub mod chat_assistant_prefill;
pub mod chat_assistant_turn;
pub mod chat_classification;
pub mod chat_grammar_trigger;
pub mod chat_message_parse_outcome;
pub mod chat_output;
pub mod chat_output_completeness;
pub mod chat_output_token;
pub mod chat_piece_role;
pub mod chat_prompt_part;
pub mod chat_session;
pub mod chat_session_options;
pub mod chat_span;
pub mod chat_span_kind;
pub mod chat_spans;
pub mod chat_template_tool_calls;
pub mod chat_templates;
pub mod chat_tool_choice;
pub mod classified_sample;
pub mod context;
pub mod cpp_text_view;
pub mod dry_sequence_breakers;
pub mod error;
pub mod eval_multimodal_chunks_params;
pub mod generation_progress;
pub mod gguf_context;
pub mod gguf_context_error;
pub mod gguf_tensor_f32;
pub mod gguf_type;
pub mod grammar_matcher;
pub mod ingest_outcome;
pub mod invalid_numa_strategy;
pub mod json_probe_outcome;
pub mod json_schema_to_grammar;
pub mod lazy_grammar_triggers;
pub mod llama_backend;
pub mod llama_backend_device;
pub mod llama_backend_device_type;
pub mod llama_backend_numa_strategy;
pub mod llama_batch;
pub mod llama_batch_entry;
pub mod llama_time_us;
pub mod llama_token_attr;
pub mod llama_token_attrs;
pub mod llama_token_attrs_from_int_error;
pub mod llguidance_sampler;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends_error;
#[cfg(feature = "dynamic-backends")]
pub mod load_backends_from_path;
pub mod log_options;
pub mod log_record;
pub mod mask_outcome;
pub mod max_devices;
pub mod model;
pub mod mtmd;
pub mod parsed_chat;
pub mod raw_chat_message;
pub mod sampled_token;
pub mod sampled_token_classifier;
pub mod sampled_token_section;
pub mod sampling;
pub mod sanitized_grammar;
pub mod send_logs_to_log;
pub mod streaming_json_probe;
pub mod token;
pub mod token_piece;
pub mod tool_call_format;

pub use error::{
    ChatSessionError, ChatTemplateError, ChatTemplatesError, DecodeError, EmbeddingsError,
    EncodeError, EvalMultimodalChunksError, FfiContractError, FfiStatusError, GrammarError,
    IngestError, JsonSchemaToGrammarError, LlamaContextLoadError, LlamaCppError,
    LlamaLoraAdapterInitError, LlamaLoraAdaptersError, LlamaModelLoadError, LogitsError,
    MemorySeqAddError, MemorySeqCpError, MemorySeqDivError, MemorySeqPosMaxError, MemorySeqRmError,
    MetaValError, ModelParamsError, NewLlamaChatMessageError, ParseChatMessageError,
    ProcessFailure, Result, SampleError, SamplerAcceptError, SamplingError, StringToTokenError,
    TokenAttrError, TokenOutsideVocabulary, TokenToStringError,
};

pub use bare_json_tool_calls::BareJsonToolCalls;
pub use chat_assistant_prefill::ChatAssistantPrefill;
pub use chat_assistant_turn::ChatAssistantTurn;
pub use chat_classification::ChatClassification;
pub use chat_grammar_trigger::ChatGrammarTrigger;
pub use chat_message_parse_outcome::ChatMessageParseOutcome;
pub use chat_output::ChatOutput;
pub use chat_output_completeness::ChatOutputCompleteness;
pub use chat_prompt_part::ChatPromptPart;
pub use chat_session::ChatSession;
pub use chat_session_options::ChatSessionOptions;
pub use chat_templates::ChatTemplates;
pub use chat_tool_choice::ChatToolChoice;
pub use classified_sample::ClassifiedSample;
pub use eval_multimodal_chunks_params::EvalMultimodalChunksParams;
pub use generation_progress::GenerationProgress;
pub use llama_backend_device::LlamaBackendDevice;
pub use llama_backend_device_type::LlamaBackendDeviceType;
pub use llama_cpp_bindings_types::{
    BracketedJsonShape, KeyValueXmlTagsShape, PairedQuoteShape, ParsedChatMessage, ParsedToolCall,
    ParsedToolCallId, TokenUsage, TokenUsageError, ToolCallArgsShape, ToolCallArguments,
    ToolCallMarkers, ToolCallValueQuote, XmlTagsShape,
};
pub use llama_cpp_gbnf::gbnf_validation_error::GbnfValidationError;
pub use raw_chat_message::RawChatMessage;
pub use sampled_token::SampledToken;
pub use sampled_token_classifier::SampledTokenClassifier;
pub use sampled_token_section::SampledTokenSection;
pub use token_piece::TokenPiece;

pub use json_schema_to_grammar::json_schema_to_grammar;
pub use llama_time_us::llama_time_us;
pub use max_devices::max_devices;

pub use log_options::LogOptions;
pub use send_logs_to_log::send_logs_to_log;
