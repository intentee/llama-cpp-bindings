use std::ffi::NulError;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::str::Utf8Error;
use std::string::FromUtf8Error;

use llama_cpp_wrapper_string::wrapper_string_error::WrapperStringError;

use crate::llama_token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelError {
    #[error("the llama.cpp sampler grew the token data array beyond its capacity")]
    CandidatesGrownBeyondCapacity { size: usize, capacity: usize },
    #[error("chat template not found - returned null pointer")]
    ChatTemplateMissing,
    #[error("null byte in string")]
    ChatTemplateNameContainsNulByte(#[source] NulError),
    #[error("the llama.cpp library ran out of memory")]
    ChatTemplatesLlamaCppOutOfMemory,
    #[error("not enough memory")]
    ChatTemplatesNotEnoughMemory,
    #[error("{message}")]
    ChatTemplatesReported { message: String },
    #[error("decision label {index} is not valid UTF-8")]
    DecisionLabelNotUtf8 {
        index: u32,
        #[source]
        source: Utf8Error,
    },
    #[error("the llama.cpp library ran out of memory")]
    DecisionTypeLlamaCppOutOfMemory,
    #[error("not enough memory")]
    DecisionTypeNotEnoughMemory,
    #[error("{message}")]
    DecisionTypeReported { message: String },
    #[error(
        "an override template must not be empty, because llama.cpp reads an empty override as no override"
    )]
    EmptyTemplateOverride,
    #[error("the llama.cpp library ran out of memory")]
    GgufInfoLlamaCppOutOfMemory,
    #[error("not enough memory")]
    GgufInfoNotEnoughMemory,
    #[error("null byte in string")]
    GgufInfoPathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    GgufInfoPathNotUtf8(PathBuf),
    #[error("{message}")]
    GgufInfoReported { message: String },
    #[error("the llama.cpp library ran out of memory")]
    LoadLlamaCppOutOfMemory,
    #[error("not enough memory")]
    LoadNotEnoughMemory,
    #[error("null byte in string")]
    LoadPathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    LoadPathNotUtf8(PathBuf),
    #[error("{message}")]
    LoadReported { message: String },
    #[error("adapter file not found: {0}")]
    LoraAdapterFileNotFound(PathBuf),
    #[error("null byte in string")]
    LoraAdapterPathContainsNulByte(#[source] NulError),
    #[error("failed to convert path {0} to str")]
    LoraAdapterPathNotUtf8(PathBuf),
    #[error("adapter could not be loaded")]
    LoraAdapterUnloadable,
    #[error("null byte in string")]
    MetaKeyContainsNulByte(#[source] NulError),
    #[error("Negative return value. Likely due to a missing index or key. Got return value: {0}")]
    MetaNegativeReturn(i32),
    #[error("FromUtf8Error")]
    MetaValueNotUtf8(#[source] FromUtf8Error),
    #[error("model file not found: {0}")]
    ModelFileNotFound(PathBuf),
    #[error("the model has no chat template")]
    ModelHasNoChatTemplate,
    #[error("model could not be loaded")]
    ModelUnloadable,
    #[error("Insufficient Buffer Space {0}")]
    PieceInsufficientBufferSpace(c_int),
    #[error("Integer conversion error")]
    PieceIntConversion(#[source] TryFromIntError),
    #[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
    PieceTokenOutsideVocabulary { token: LlamaToken, n_vocab: i32 },
    #[error("the llama.cpp library ran out of memory")]
    RenderLlamaCppOutOfMemory,
    #[error("not enough memory")]
    RenderNotEnoughMemory,
    #[error("{message}")]
    RenderReported { message: String },
    #[error("the llama.cpp sampler selected a token outside the token data array")]
    SamplerSelectedCandidateOutsideCandidates {
        selected: i64,
        candidate_count: usize,
    },
    #[error("the token data array selects a token outside itself")]
    SelectedCandidateOutsideCandidates {
        selected: usize,
        candidate_count: usize,
    },
    #[error("template could not be executed: {message}")]
    TemplateExecutionFailed { message: String },
    #[error("template globals contain an interior NUL byte")]
    TemplateGlobalsContainNulByte(#[source] NulError),
    #[error("template globals are not valid JSON: {message}")]
    TemplateGlobalsInvalidJson { message: String },
    #[error("template globals are not a JSON object")]
    TemplateGlobalsNotAJsonObject,
    #[error("template is invalid: {message}")]
    TemplateInvalid { message: String },
    #[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
    TokenAttributesTokenOutsideVocabulary { token: LlamaToken, n_vocab: i32 },
    #[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
    TokenTextTokenOutsideVocabulary { token: LlamaToken, n_vocab: i32 },
    #[error("a tokenizer length or count does not fit its integer type")]
    TokenizeIntConversion(#[source] TryFromIntError),
    #[error("the llama.cpp library ran out of memory")]
    TokenizeLlamaCppOutOfMemory,
    #[error("not enough memory")]
    TokenizeNotEnoughMemory,
    #[error("{message}")]
    TokenizeReported { message: String },
    #[error("the text to tokenize contains a NUL byte")]
    TokenizeTextContainsNulByte(#[source] NulError),
    #[error("decision type {0} is not a value the llama.cpp wrapper defines")]
    UnknownDecisionType(llama_cpp_bindings_sys::llama_rs_decision_type),
    #[error("rope type {0} is not a value llama.cpp defines")]
    UnknownRopeType(llama_cpp_bindings_sys::llama_rope_type),
    #[error("Unknown Value {0}")]
    UnknownTokenAttributes(c_uint),
    #[error("Unknown Value {0}")]
    UnknownVocabType(llama_cpp_bindings_sys::llama_vocab_type),
    #[error("{operation} returned unknown FFI status {code}")]
    UnknownWrapperStatus { operation: &'static str, code: i64 },
    #[error("{operation} returned an invalid FFI response: was given a null {argument} argument")]
    WrapperRefusedNullArgument {
        operation: &'static str,
        argument: &'static str,
    },
    #[error(transparent)]
    WrapperString(#[from] WrapperStringError),
    #[error(
        "{operation} returned an invalid FFI response: success status contained a null {output}"
    )]
    WrapperSucceededWithoutOutput {
        operation: &'static str,
        output: &'static str,
    },
}
