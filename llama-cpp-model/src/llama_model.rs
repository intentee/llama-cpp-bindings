use encoding_rs::CoderResult;
use once_cell::sync::OnceCell;
use std::cmp;
use std::ffi::c_int;
use std::ffi::{CStr, CString, c_char};
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::num::NonZeroU16;
use std::num::TryFromIntError;
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;
use std::sync::Arc;
use toktrie::ApproximateTokEnv;
use toktrie::INVALID_TOKEN;
use toktrie::TokRxInfo;
use toktrie::TokTrie;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

use crate::add_bos::AddBos;
use crate::chat_templates::ChatTemplates;
use crate::decision_type::DecisionType;
use crate::llama_chat_template::LlamaChatTemplate;
use crate::llama_lora_adapter::LlamaLoraAdapter;
use crate::llama_token::LlamaToken;
use crate::llama_token_attrs::LlamaTokenAttrs;
use crate::model_error::ModelError;
use crate::parse_special_tokens::ParseSpecialTokens;
use crate::path_c_string::path_c_string;
use crate::piece_write::PieceWrite;
use crate::rope_type::RopeType;
use crate::sampled_token::SampledToken;
use crate::tokenizer_input::TokenizerInput;
use crate::vocab_type::VocabType;
use crate::vocabulary_token::VocabularyToken;

const PIECE_PROBE_LEN: c_int = 32;

const PIECE_PROBE_SIZE: usize = PIECE_PROBE_LEN.unsigned_abs() as usize;

fn checked_tokenizer_length(length: usize) -> Result<c_int, ModelError> {
    c_int::try_from(length).map_err(ModelError::TokenizeIntConversion)
}

fn cstring_with_validated_len(text: &str) -> Result<TokenizerInput, ModelError> {
    let length = checked_tokenizer_length(text.len())?;
    let c_string = CString::new(text).map_err(ModelError::TokenizeTextContainsNulByte)?;

    Ok(TokenizerInput {
        text: c_string,
        length,
    })
}

const fn present_special_token(
    token_id: llama_cpp_bindings_sys::llama_token,
) -> Option<LlamaToken> {
    if token_id == llama_cpp_bindings_sys::LLAMA_TOKEN_NULL {
        None
    } else {
        Some(LlamaToken(token_id))
    }
}

fn decode_piece(decoder: &mut encoding_rs::Decoder, bytes: &[u8]) -> String {
    let mut piece = String::with_capacity(bytes.len());
    let mut undecoded = bytes;

    loop {
        let (result, read, _had_replacements) =
            decoder.decode_to_string(undecoded, &mut piece, false);

        undecoded = undecoded.split_at(read).1;

        match result {
            CoderResult::InputEmpty => return piece,
            CoderResult::OutputFull => piece.reserve(char::MAX_LEN_UTF8),
        }
    }
}

/// # Safety
///
/// `out_model` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_load_model_from_file` call (or null); `out_error` is read and freed only in the
/// CXX-exception arm.
unsafe fn load_model_from_file_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_load_model_from_file_status,
    out_model: *mut llama_cpp_bindings_sys::llama_model,
    out_error: *mut c_char,
    path: &Path,
) -> Result<LlamaModel, ModelError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_OK => {
            let model = NonNull::new(out_model).ok_or_else(|| {
                ModelError::from(FfiError::ContractViolated {
                    operation: "llama_rs_load_model_from_file",
                    detail: "success status contained a null model",
                })
            })?;
            let vocab = unsafe { llama_cpp_bindings_sys::llama_model_get_vocab(model.as_ptr()) };

            Ok(LlamaModel {
                model,
                vocab,
                n_vocab: unsafe { llama_cpp_bindings_sys::llama_vocab_n_tokens(vocab) },
                tok_env: OnceCell::new(),
                chat_templates: OnceCell::new(),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_RETURNED_NULL => {
            if path.exists() {
                Err(ModelError::ModelUnloadable)
            } else {
                Err(ModelError::ModelFileNotFound(path.to_path_buf()))
            }
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::LoadNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::LoadLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_load_model_from_file",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(ModelError::LoadReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_PATH_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_load_model_from_file",
                detail: "was given a null path argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_MODEL_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_load_model_from_file",
                detail: "was given a null out_model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_ERROR_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_load_model_from_file",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(FfiError::UnknownStatus {
            operation: "llama_rs_load_model_from_file",
            code: i64::from(other),
        }
        .into()),
    }
}

/// # Safety
///
/// `out_error` must be the pointer populated by the preceding `llama_rs_model_decision_type` call
/// (or null); it is read and freed only in the CXX-exception arm.
unsafe fn model_decision_type_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_model_decision_type_status,
    out_decision_type: llama_cpp_bindings_sys::llama_rs_decision_type,
    out_error: *mut c_char,
) -> Result<DecisionType, ModelError> {
    let contract_violation = |detail: &'static str| -> Result<DecisionType, ModelError> {
        Err(FfiError::ContractViolated {
            operation: "llama_rs_model_decision_type",
            detail,
        }
        .into())
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_OK => {
            DecisionType::try_from(out_decision_type)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::DecisionTypeNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::DecisionTypeLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_model_decision_type",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(ModelError::DecisionTypeReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_NULL_MODEL_ARG => {
            contract_violation("was given a null model argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_NULL_OUT_DECISION_TYPE_ARG => {
            contract_violation("was given a null out_decision_type argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_NULL_OUT_ERROR_ARG => {
            contract_violation("was given a null out_error argument")
        }
        other => Err(FfiError::UnknownStatus {
            operation: "llama_rs_model_decision_type",
            code: i64::from(other),
        }
        .into()),
    }
}

/// # Safety
///
/// Every non-null pointer `read_label` returns must point to a null-terminated C string that
/// stays valid until this function returns.
unsafe fn collect_decision_labels(
    read_label: impl Fn(u32) -> *const c_char,
) -> Result<Vec<String>, ModelError> {
    (0..=u32::MAX)
        .map_while(|index| {
            let label = read_label(index);

            if label.is_null() {
                return None;
            }

            Some(
                unsafe { CStr::from_ptr(label) }
                    .to_str()
                    .map(str::to_owned)
                    .map_err(|source| ModelError::DecisionLabelNotUtf8 { index, source }),
            )
        })
        .collect()
}

fn initialize_lora_adapter<'model>(
    initialize: impl FnOnce() -> *mut llama_cpp_bindings_sys::llama_adapter_lora,
) -> Result<LlamaLoraAdapter<'model>, ModelError> {
    let adapter = NonNull::new(initialize()).ok_or(ModelError::LoraAdapterUnloadable)?;
    Ok(unsafe { LlamaLoraAdapter::new(adapter) })
}

fn build_approximate_tok_env(model: &LlamaModel) -> Result<Arc<ApproximateTokEnv>, ModelError> {
    let n_vocab = model.n_vocab().cast_unsigned();
    let end_of_generation_token = present_special_token(unsafe {
        llama_cpp_bindings_sys::llama_vocab_eot(model.vocab_ptr())
    })
    .or_else(|| model.token_eos());
    let tok_eos = match end_of_generation_token {
        Some(LlamaToken(token_id)) => token_id.cast_unsigned(),
        None => INVALID_TOKEN,
    };
    let info = TokRxInfo::new(n_vocab, tok_eos);

    let mut words = Vec::with_capacity(n_vocab as usize);

    for token_id in 0..n_vocab.cast_signed() {
        let token = LlamaToken(token_id);
        let bytes = model.read_piece(token, false, None, <[u8]>::to_vec)?;
        if bytes.is_empty() {
            let special_bytes = model.read_piece(token, true, None, <[u8]>::to_vec)?;
            if special_bytes.is_empty() {
                words.push(vec![]);
            } else {
                let mut marked = Vec::with_capacity(special_bytes.len().saturating_add(1));
                marked.push(0xFF);
                marked.extend(special_bytes);
                words.push(marked);
            }
        } else {
            words.push(bytes);
        }
    }

    let trie = TokTrie::from(&info, &words);
    Ok(Arc::new(ApproximateTokEnv::new(trie)))
}

/// # Safety
///
/// `out_error` must be the pointer populated by the preceding `llama_rs_tokenize` call (or
/// null when no error was produced); it is read and freed only in the CXX-exception arm.
unsafe fn tokenize_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_tokenize_status,
    out_count: c_int,
    out_error: *mut c_char,
) -> Result<c_int, ModelError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ModelError::TokenizeNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ModelError::TokenizeLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_tokenize",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(ModelError::TokenizeReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_VOCAB_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(FfiError::ContractViolated {
                operation: "llama_rs_tokenize",
                detail: "was given a null vocab argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_TEXT_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(FfiError::ContractViolated {
                operation: "llama_rs_tokenize",
                detail: "was given a null text argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_RETURNED_COUNT_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(FfiError::ContractViolated {
                operation: "llama_rs_tokenize",
                detail: "was given a null out_returned_count argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_ERROR_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(FfiError::ContractViolated {
                operation: "llama_rs_tokenize",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(FfiError::UnknownStatus {
                operation: "llama_rs_tokenize",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn invoke_rs_tokenize(
    vocab: *const llama_cpp_bindings_sys::llama_vocab,
    text: *const c_char,
    text_len: c_int,
    tokens: *mut llama_cpp_bindings_sys::llama_token,
    n_tokens_max: c_int,
    add_bos: bool,
    parse_special: bool,
) -> Result<c_int, ModelError> {
    let mut out_count: i32 = 0;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_tokenize(
            vocab,
            text,
            text_len,
            tokens,
            n_tokens_max,
            add_bos,
            parse_special,
            &raw mut out_count,
            &raw mut out_error,
        )
    };
    unsafe { tokenize_status_to_result(status, out_count, out_error) }
}

fn checked_token_count(size: i32) -> Result<usize, ModelError> {
    usize::try_from(size).map_err(ModelError::TokenizeIntConversion)
}

fn tokenize_into_buffer(
    estimated_capacity: usize,
    invoke: impl Fn(*mut llama_cpp_bindings_sys::llama_token, c_int) -> Result<c_int, ModelError>,
) -> Result<Vec<LlamaToken>, ModelError> {
    let buffer_capacity = checked_tokenizer_length(estimated_capacity)?;
    let mut buffer: Vec<LlamaToken> = Vec::with_capacity(estimated_capacity);

    let size = invoke(
        buffer
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token>(),
        buffer_capacity,
    )?;

    let size = if size.is_negative() {
        let required_size = size.wrapping_neg();

        buffer.reserve_exact(checked_token_count(required_size)?);
        invoke(
            buffer
                .as_mut_ptr()
                .cast::<llama_cpp_bindings_sys::llama_token>(),
            required_size,
        )?
    } else {
        size
    };

    let size = checked_token_count(size)?;

    unsafe { buffer.set_len(size) }

    Ok(buffer)
}

fn extract_meta_string<TCFunction>(
    c_function: TCFunction,
    capacity: usize,
) -> Result<String, ModelError>
where
    TCFunction: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];
    let result = c_function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len());

    if result < 0 {
        return Err(ModelError::MetaNegativeReturn(result));
    }

    let returned_len = result.cast_unsigned() as usize;

    if returned_len >= capacity {
        return extract_meta_string(c_function, returned_len + 1);
    }

    if buffer.get(returned_len) != Some(&0) {
        return Err(ModelError::MetaNegativeReturn(-1));
    }

    buffer.truncate(returned_len);

    String::from_utf8(buffer).map_err(ModelError::MetaValueNotUtf8)
}

pub struct LlamaModel {
    pub model: NonNull<llama_cpp_bindings_sys::llama_model>,
    vocab: *const llama_cpp_bindings_sys::llama_vocab,
    n_vocab: i32,
    tok_env: OnceCell<Arc<ApproximateTokEnv>>,
    chat_templates: OnceCell<ChatTemplates>,
}

impl Debug for LlamaModel {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlamaModel")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// # Errors
    ///
    /// See [`ModelError`] for more information.
    ///
    /// # Panics
    ///
    /// Panics if a valid UTF-8 path somehow contains interior null bytes (should never happen).
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        LlamaModelParams { params, .. }: &LlamaModelParams,
    ) -> Result<Self, ModelError> {
        let path = path.as_ref();
        let cstr = path_c_string(path)?;

        if !path.exists() {
            return Err(ModelError::ModelFileNotFound(path.to_path_buf()));
        }
        let mut out_model: *mut llama_cpp_bindings_sys::llama_model = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_load_model_from_file(
                cstr.as_ptr(),
                *params,
                &raw mut out_model,
                &raw mut out_error,
            )
        };
        unsafe { load_model_from_file_status_to_result(status, out_model, out_error, path) }
    }

    #[must_use]
    pub fn adds_bos_token(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_vocab_get_add_bos(self.vocab_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`ModelError::ChatTemplateNameContainsNulByte`] when `name` contains a NUL byte,
    /// or [`ModelError::ChatTemplateMissing`] when the model has no chat template by that name.
    pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ModelError> {
        let name_cstr = name
            .map(CString::new)
            .transpose()
            .map_err(ModelError::ChatTemplateNameContainsNulByte)?;
        let name_ptr = name_cstr.as_ref().map_or(ptr::null(), |name| name.as_ptr());
        let result = unsafe {
            llama_cpp_bindings_sys::llama_model_chat_template(self.model.as_ptr(), name_ptr)
        };

        if result.is_null() {
            Err(ModelError::ChatTemplateMissing)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };

            Ok(LlamaChatTemplate(chat_template_cstr.to_owned()))
        }
    }

    #[must_use]
    pub fn classifier_free_guidance(&self) -> bool {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_model_classifier_free_guidance(self.model.as_ptr())
        }
    }

    /// # Errors
    ///
    /// Returns [`ModelError::DecisionLabelNotUtf8`] when a label is not valid UTF-8.
    pub fn decision_labels(&self) -> Result<Vec<String>, ModelError> {
        unsafe {
            collect_decision_labels(|index| {
                llama_cpp_bindings_sys::llama_rs_model_decision_label(self.model.as_ptr(), index)
            })
        }
    }

    /// # Errors
    ///
    /// Returns [`ModelError`] when the wrapper fails or reports a decision type it does
    /// not define.
    pub fn decision_type(&self) -> Result<DecisionType, ModelError> {
        let mut out_decision_type = llama_cpp_bindings_sys::LLAMA_RS_DECISION_TYPE_NONE;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_model_decision_type(
                self.model.as_ptr(),
                &raw mut out_decision_type,
                &raw mut out_error,
            )
        };

        unsafe { model_decision_type_status_to_result(status, out_decision_type, out_error) }
    }

    #[must_use]
    pub fn decode_start_token(&self) -> Option<LlamaToken> {
        present_special_token(unsafe {
            llama_cpp_bindings_sys::llama_model_decoder_start_token(self.model.as_ptr())
        })
    }

    #[must_use]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_has_encoder(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn is_eog_token(&self, token: &SampledToken) -> bool {
        let (SampledToken::Content(LlamaToken(id))
        | SampledToken::Reasoning(LlamaToken(id))
        | SampledToken::ToolCall(LlamaToken(id))
        | SampledToken::Undeterminable(LlamaToken(id))) = *token;

        unsafe { llama_cpp_bindings_sys::llama_vocab_is_eog(self.vocab_ptr(), id) }
    }

    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_hybrid(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// See [`ModelError`] for more information.
    pub fn lora_adapter_init(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter<'_>, ModelError> {
        let path = path.as_ref();

        let cstr = llama_cpp_path_c_string::path_c_string::path_c_string(path).map_err(
            |error| match error {
                PathCStringError::ContainsNulByte(source) => {
                    ModelError::LoraAdapterPathContainsNulByte(source)
                }
                PathCStringError::NotUtf8(path) => ModelError::LoraAdapterPathNotUtf8(path),
            },
        )?;

        if !path.exists() {
            return Err(ModelError::LoraAdapterFileNotFound(path.to_path_buf()));
        }
        initialize_lora_adapter(|| unsafe {
            llama_cpp_bindings_sys::llama_adapter_lora_init(self.model.as_ptr(), cstr.as_ptr())
        })
    }

    #[must_use]
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// # Errors
    /// Returns an error if the index is out of range or the key is not valid UTF-8.
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, ModelError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_bindings_sys::llama_model_meta_key_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// # Errors
    /// Returns an error if the key is not found or the value is not valid UTF-8.
    pub fn meta_val_str(&self, key: &str) -> Result<String, ModelError> {
        let key_cstring = CString::new(key).map_err(ModelError::MetaKeyContainsNulByte)?;
        let key_ptr = key_cstring.as_ptr();

        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_bindings_sys::llama_model_meta_val_str(
                    self.model.as_ptr(),
                    key_ptr,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// # Errors
    /// Returns an error if the index is out of range or the value is not valid UTF-8.
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, ModelError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_bindings_sys::llama_model_meta_val_str_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    #[must_use]
    pub fn n_cls_out(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_model_n_cls_out(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the value returned by llama.cpp does not fit into a `u32`.
    pub fn n_ctx_train(&self) -> Result<u32, TryFromIntError> {
        let n_ctx_train =
            unsafe { llama_cpp_bindings_sys::llama_model_n_ctx_train(self.model.as_ptr()) };

        u32::try_from(n_ctx_train)
    }

    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_bindings_sys::llama_model_n_embd(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn n_embd_out(&self) -> c_int {
        unsafe { llama_cpp_bindings_sys::llama_model_n_embd_out(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head(&self) -> Result<u32, TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head(self.model.as_ptr()) })
    }

    /// # Errors
    ///
    /// Returns an error if the KV head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head_kv(&self) -> Result<u32, TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head_kv(self.model.as_ptr()) })
    }

    /// # Errors
    ///
    /// Returns an error if the layer count returned by llama.cpp does not fit into a `u32`.
    pub fn n_layer(&self) -> Result<u32, TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_layer(self.model.as_ptr()) })
    }

    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_n_params(self.model.as_ptr()) }
    }

    #[must_use]
    pub const fn n_vocab(&self) -> i32 {
        self.n_vocab
    }

    /// # Errors
    ///
    /// Returns [`ModelError::UnknownRopeType`] when llama.cpp reports a rope type it does not
    /// define.
    pub fn rope_type(&self) -> Result<RopeType, ModelError> {
        RopeType::try_from(unsafe {
            llama_cpp_bindings_sys::llama_rs_model_rope_type(self.model.as_ptr())
        })
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_size(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// - if [`str`] contains a null byte
    /// - if an integer conversion fails during tokenization
    pub fn str_to_token(
        &self,
        str: &str,
        add_bos: AddBos,
        parse_special_tokens: ParseSpecialTokens,
    ) -> Result<Vec<LlamaToken>, ModelError> {
        let add_bos = match add_bos {
            AddBos::Always => true,
            AddBos::Never => false,
        };
        let parse_special = match parse_special_tokens {
            ParseSpecialTokens::Always => true,
            ParseSpecialTokens::Never => false,
        };

        let tokens_estimation = cmp::max(8, (str.len() / 2).saturating_add(usize::from(add_bos)));
        let TokenizerInput {
            text: c_string,
            length: c_string_len,
        } = cstring_with_validated_len(str)?;
        let vocab = self.vocab_ptr();

        tokenize_into_buffer(tokens_estimation, |tokens, n_tokens_max| {
            invoke_rs_tokenize(
                vocab,
                c_string.as_ptr(),
                c_string_len,
                tokens,
                n_tokens_max,
                add_bos,
                parse_special,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`ModelError::TokenOutsideVocabulary`] when the token is not part of this
    /// model's vocabulary, or [`ModelError::UnknownTokenAttributes`] when llama.cpp reports
    /// attributes this library does not know.
    pub fn token_attr(&self, token: LlamaToken) -> Result<LlamaTokenAttrs, ModelError> {
        let LlamaToken(id) = self.require_vocabulary_token(token)?;
        let token_attr =
            unsafe { llama_cpp_bindings_sys::llama_vocab_get_attr(self.vocab_ptr(), id) };

        LlamaTokenAttrs::try_from(token_attr)
    }

    #[must_use]
    pub fn token_bos(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_bos(self.vocab_ptr()) })
    }

    #[must_use]
    pub fn token_eos(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_eos(self.vocab_ptr()) })
    }

    #[must_use]
    pub fn token_mask(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_mask(self.vocab_ptr()) })
    }

    #[must_use]
    pub fn token_nl(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_nl(self.vocab_ptr()) })
    }

    #[must_use]
    pub fn token_sep(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_sep(self.vocab_ptr()) })
    }

    /// # Errors
    ///
    /// Returns [`ModelError::TokenOutsideVocabulary`] when the token is not part of this model's
    /// vocabulary.
    pub fn token_text_bytes(&self, token: LlamaToken) -> Result<Vec<u8>, ModelError> {
        let LlamaToken(id) = self.require_vocabulary_token(token)?;
        let text = unsafe { llama_cpp_bindings_sys::llama_vocab_get_text(self.vocab_ptr(), id) };

        Ok(unsafe { CStr::from_ptr(text) }.to_bytes().to_vec())
    }

    /// # Errors
    ///
    /// - if the token type is unknown
    ///
    /// - if the returned size from llama.cpp does not fit into a `usize`
    pub fn token_to_piece(
        &self,
        token: SampledToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, ModelError> {
        self.read_piece(token.token(), special, lstrip, |piece| {
            decode_piece(decoder, piece)
        })
    }

    /// # Errors
    ///
    /// - if the token is not part of this model's vocabulary
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    /// - if an integer conversion fails
    pub fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, ModelError> {
        let token = self.require_vocabulary_token(token)?;
        let buffer_len = c_int::try_from(buffer_size).map_err(ModelError::PieceIntConversion)?;
        let mut buffer: Vec<u8> = vec![0u8; buffer_size];
        match self.write_piece(token, &mut buffer, buffer_len, special, lstrip) {
            PieceWrite::Written(written) => {
                buffer.truncate(written);

                Ok(buffer)
            }
            PieceWrite::NeedsBuffer(negated_size) => {
                Err(ModelError::PieceInsufficientBufferSpace(negated_size))
            }
        }
    }

    /// Every token of the vocabulary, in id order, with the text it decodes to on its own.
    pub fn tokens(
        &self,
        decode_special: bool,
    ) -> impl Iterator<Item = Result<VocabularyToken, ModelError>> + '_ {
        (0..self.n_vocab()).map(LlamaToken::new).map(move |token| {
            self.token_to_piece(
                SampledToken::Content(token),
                &mut encoding_rs::UTF_8.new_decoder(),
                decode_special,
                None,
            )
            .map(|piece| VocabularyToken { token, piece })
        })
    }

    #[must_use]
    pub const fn vocab_ptr(&self) -> *const llama_cpp_bindings_sys::llama_vocab {
        self.vocab
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp emits a vocab type that is not known to this library.
    pub fn vocab_type(&self) -> Result<VocabType, ModelError> {
        let vocab_type = unsafe { llama_cpp_bindings_sys::llama_vocab_type(self.vocab_ptr()) };

        VocabType::try_from(vocab_type)
    }

    fn read_piece<TOutput>(
        &self,
        token: LlamaToken,
        special: bool,
        lstrip: Option<NonZeroU16>,
        read: impl FnOnce(&[u8]) -> TOutput,
    ) -> Result<TOutput, ModelError> {
        let token = self.require_vocabulary_token(token)?;
        let mut probe = [0_u8; PIECE_PROBE_SIZE];

        match self.write_piece(token, &mut probe, PIECE_PROBE_LEN, special, lstrip) {
            PieceWrite::Written(written) => Ok(read(probe.split_at(written).0)),
            PieceWrite::NeedsBuffer(negated_size) => self
                .token_to_piece_bytes(token, negated_size.unsigned_abs() as usize, special, lstrip)
                .map(|piece| read(&piece)),
        }
    }

    const fn require_vocabulary_token(&self, token: LlamaToken) -> Result<LlamaToken, ModelError> {
        if token.is_within_vocabulary(self.n_vocab()) {
            Ok(token)
        } else {
            Err(ModelError::TokenOutsideVocabulary {
                token,
                n_vocab: self.n_vocab(),
            })
        }
    }

    fn write_piece(
        &self,
        LlamaToken(token_id): LlamaToken,
        buffer: &mut [u8],
        buffer_len: c_int,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> PieceWrite {
        PieceWrite::from(unsafe {
            llama_cpp_bindings_sys::llama_rs_token_to_piece(
                self.vocab,
                token_id,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer_len,
                lstrip.map_or(0, |strip_count| i32::from(strip_count.get())),
                special,
            )
        })
    }
}

impl LlamaModel {
    /// # Errors
    /// Returns [`ModelError`] when a token's byte piece cannot be
    /// retrieved. The legitimate "this token has no byte piece" case is treated
    /// as empty (not an error); a piece that overflows the probe buffer is
    /// re-read at the exact size rather than dropped.
    pub fn approximate_tok_env(&self) -> Result<Arc<ApproximateTokEnv>, ModelError> {
        self.tok_env
            .get_or_try_init(|| build_approximate_tok_env(self))
            .map(Arc::clone)
    }

    /// Returns the model's chat templates, preparing them on first use.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the model has no chat template or llama.cpp cannot prepare
    /// it.
    pub fn chat_templates(&self) -> Result<&ChatTemplates, ModelError> {
        self.chat_templates
            .get_or_try_init(|| unsafe { ChatTemplates::new(self.model) })
    }

    /// `template` in place of the model's own chat templates, the way llama.cpp serves a model
    /// whose template cannot render a request, such as tool calls.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::EmptyTemplateOverride`] for an empty `template`, or another
    /// [`ModelError`] when llama.cpp cannot prepare the template.
    pub fn chat_templates_with_override(
        &self,
        template: &LlamaChatTemplate,
    ) -> Result<ChatTemplates, ModelError> {
        unsafe { ChatTemplates::with_override(self.model, template) }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_model_free(self.model.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ffi::CStr;
    use std::ffi::CString;
    use std::ffi::c_char;
    use std::mem::ManuallyDrop;
    use std::path::Path;
    use std::ptr;
    use std::slice;
    use std::str;

    use llama_cpp_ffi_status::ffi_error::FfiError;

    use super::checked_token_count;
    use super::checked_tokenizer_length;
    use super::cstring_with_validated_len;
    use super::extract_meta_string;
    use super::initialize_lora_adapter;
    use super::load_model_from_file_status_to_result;
    use super::model_decision_type_status_to_result;
    use super::tokenize_into_buffer;

    use super::collect_decision_labels;
    use super::tokenize_status_to_result;
    use crate::decision_type::DecisionType;
    use crate::model_error::ModelError;

    const NOT_A_DECISION_MODEL: llama_cpp_bindings_sys::llama_rs_decision_type =
        llama_cpp_bindings_sys::LLAMA_RS_DECISION_TYPE_NONE;

    fn label_at(labels: &[&CStr], index: u32) -> *const c_char {
        labels
            .get(index as usize)
            .map_or(ptr::null(), |label| label.as_ptr())
    }

    #[test]
    fn cstring_with_validated_len_reports_a_nul_byte_in_the_text() {
        assert_eq!(
            cstring_with_validated_len("null\0byte").err(),
            CString::new("null\0byte")
                .err()
                .map(ModelError::TokenizeTextContainsNulByte)
        );
    }

    #[test]
    fn a_text_longer_than_a_c_int_is_refused_before_it_is_copied() {
        let too_long = vec![0_u8; usize::try_from(i32::MAX).unwrap() + 1];

        assert_eq!(
            cstring_with_validated_len(str::from_utf8(&too_long).unwrap()).err(),
            i32::try_from(too_long.len())
                .err()
                .map(ModelError::TokenizeIntConversion)
        );
    }

    #[test]
    fn checked_tokenizer_length_in_range_returns_value() {
        assert_eq!(checked_tokenizer_length(8), Ok(8));
    }

    #[test]
    fn checked_token_count_non_negative_returns_value() {
        assert_eq!(checked_token_count(5), Ok(5));
    }

    #[test]
    fn tokenize_into_buffer_single_pass_sets_length() {
        let buffer = tokenize_into_buffer(8, |_tokens, _n_tokens_max| Ok(3)).unwrap();

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn tokenize_into_buffer_grows_buffer_when_first_pass_reports_negative_size() {
        let call_count = Cell::new(0);
        let buffer = tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 { Ok(-20) } else { Ok(15) }
        })
        .unwrap();

        assert_eq!(buffer.len(), 15);
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn tokenize_into_buffer_propagates_invocation_error() {
        let result = tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            Err(ModelError::TokenizeNotEnoughMemory)
        });

        assert_eq!(result, Err(ModelError::TokenizeNotEnoughMemory));
    }

    #[test]
    fn tokenize_into_buffer_propagates_second_invocation_error() {
        let call_count = Cell::new(0);
        let result = tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 {
                Ok(-20)
            } else {
                Err(ModelError::TokenizeNotEnoughMemory)
            }
        });

        assert_eq!(result, Err(ModelError::TokenizeNotEnoughMemory));
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn tokenize_into_buffer_refuses_a_capacity_beyond_a_c_int_before_allocating() {
        assert_eq!(
            tokenize_into_buffer(usize::MAX, |_tokens, _n_tokens_max| Ok(0)),
            Err(ModelError::TokenizeIntConversion(
                i32::try_from(usize::MAX).unwrap_err()
            ))
        );
    }

    #[test]
    fn tokenize_into_buffer_refuses_a_required_size_it_cannot_negate() {
        assert_eq!(
            tokenize_into_buffer(8, |_tokens, _n_tokens_max| Ok(i32::MIN)),
            Err(ModelError::TokenizeIntConversion(
                usize::try_from(i32::MIN).unwrap_err()
            ))
        );
    }

    #[test]
    fn tokenize_into_buffer_negative_final_size_returns_conversion_error() {
        let call_count = Cell::new(0);
        let result = tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 { Ok(-20) } else { Ok(-5) }
        });

        assert_eq!(
            result.unwrap_err(),
            ModelError::TokenizeIntConversion(usize::try_from(-5i32).unwrap_err())
        );
    }

    #[test]
    fn a_value_without_a_nul_terminator_is_reported_as_a_negative_return() {
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let buffer = unsafe { slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                buffer.get_mut(..3).unwrap().copy_from_slice(b"abc");
                2
            },
            4,
        );

        assert_eq!(result.unwrap_err(), ModelError::MetaNegativeReturn(-1));
    }

    #[test]
    fn a_negative_return_value_is_reported_as_is() {
        let result = extract_meta_string(|_buf_ptr, _buf_len| -5, 4);

        assert_eq!(result.unwrap_err(), ModelError::MetaNegativeReturn(-5));
    }

    #[test]
    fn invalid_utf8_metadata_is_reported_as_a_utf8_error() {
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let buffer = unsafe { slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                buffer
                    .get_mut(..3)
                    .unwrap()
                    .copy_from_slice(&[0xFF, 0xFE, 0]);
                2
            },
            4,
        );

        let invalid_utf8 = String::from_utf8(vec![0xFF, 0xFE]).unwrap_err();

        assert_eq!(result, Err(ModelError::MetaValueNotUtf8(invalid_utf8)));
    }

    #[test]
    fn a_value_longer_than_the_buffer_is_reread_into_a_larger_buffer() {
        let initial_capacity: usize = 4;
        let length_exceeding_initial_capacity = 10;
        let written_length = 2;
        let call_count = Cell::new(0);
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let count = call_count.get();
                call_count.set(count + 1);
                if count == 0 {
                    length_exceeding_initial_capacity
                } else {
                    let buffer =
                        unsafe { slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                    buffer
                        .get_mut(..3)
                        .unwrap()
                        .copy_from_slice(&[b'h', b'i', 0]);
                    written_length
                }
            },
            initial_capacity,
        );

        assert_eq!(result, Ok("hi".to_owned()));
    }

    #[test]
    fn lora_adapter_initialization_maps_a_null_pointer_to_unloadable() {
        let result = initialize_lora_adapter(ptr::null_mut);

        assert_eq!(result.unwrap_err(), ModelError::LoraAdapterUnloadable);
    }

    #[test]
    fn lora_adapter_initialization_owns_a_valid_pointer() {
        let pointer = ptr::NonNull::dangling();
        let adapter = ManuallyDrop::new(
            initialize_lora_adapter(|| pointer.as_ptr())
                .expect("a non-null adapter pointer must initialize"),
        );

        assert_eq!(adapter.as_ptr(), pointer.as_ptr());
    }

    #[test]
    fn load_model_success_with_null_model_is_contract_error() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_OK,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(
            result.unwrap_err(),
            ModelError::Ffi(FfiError::ContractViolated {
                operation: "llama_rs_load_model_from_file",
                detail: "success status contained a null model",
            })
        );
    }

    #[test]
    fn load_model_from_file_llama_cpp_returned_null_for_missing_path_is_file_not_found() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_RETURNED_NULL,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/definitely/missing/model.gguf"),
            )
        };

        assert_eq!(
            result.unwrap_err(),
            ModelError::ModelFileNotFound(
                Path::new("/definitely/missing/model.gguf").to_path_buf()
            )
        );
    }

    #[test]
    fn load_model_from_file_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(result.unwrap_err(), ModelError::LoadNotEnoughMemory);
    }

    #[test]
    fn load_model_from_file_cxx_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"model loading threw".as_ptr()) };
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
                Path::new("/some/path"),
            )
        };

        assert_eq!(
            result.err(),
            Some(ModelError::LoadReported {
                message: "model loading threw".to_owned()
            })
        );
    }

    #[test]
    fn load_model_from_file_cxx_exception_without_a_message_is_a_contract_error() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(
            result.unwrap_err(),
            FfiError::ContractViolated {
                operation: "llama_rs_load_model_from_file",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn load_model_from_file_unknown_status_is_preserved() {
        let result = unsafe {
            load_model_from_file_status_to_result(
                255,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/some/path"),
            )
        };

        assert_eq!(
            result.unwrap_err(),
            ModelError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_load_model_from_file",
                code: 255,
            })
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_reports_a_null_path_arg_as_a_contract_violation() {
        let outcome = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_PATH_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/missing-for-contract-test.gguf"),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_load_model_from_file",
                    detail: "was given a null path argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_reports_a_null_out_model_arg_as_a_contract_violation()
    {
        let outcome = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_MODEL_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/missing-for-contract-test.gguf"),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_load_model_from_file",
                    detail: "was given a null out_model argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_reports_a_null_out_error_arg_as_a_contract_violation()
    {
        let outcome = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_ERROR_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/missing-for-contract-test.gguf"),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_load_model_from_file",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/missing-for-contract-test.gguf"),
            )
        };
        assert_eq!(outcome.err(), Some(ModelError::LoadLlamaCppOutOfMemory));
    }

    #[test]
    fn the_decision_type_the_wrapper_reports_is_returned() {
        assert_eq!(
            unsafe {
                model_decision_type_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_OK,
                    llama_cpp_bindings_sys::LLAMA_RS_DECISION_TYPE_KEV,
                    ptr::null_mut(),
                )
            },
            Ok(DecisionType::Kev)
        );
    }

    #[test]
    fn a_decision_type_the_wrapper_does_not_define_is_refused() {
        assert_eq!(
            unsafe {
                model_decision_type_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_OK,
                    9999,
                    ptr::null_mut(),
                )
            },
            Err(ModelError::UnknownDecisionType(9999))
        );
    }

    #[test]
    fn reading_the_decision_type_reports_each_null_argument_as_a_contract_violation() {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_NULL_MODEL_ARG,
                "was given a null model argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_NULL_OUT_DECISION_TYPE_ARG,
                "was given a null out_decision_type argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                unsafe {
                    model_decision_type_status_to_result(
                        status,
                        NOT_A_DECISION_MODEL,
                        ptr::null_mut(),
                    )
                },
                Err(ModelError::Ffi(FfiError::ContractViolated {
                    operation: "llama_rs_model_decision_type",
                    detail,
                }))
            );
        }
    }

    #[test]
    fn reading_the_decision_type_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                model_decision_type_status_to_result(status, NOT_A_DECISION_MODEL, ptr::null_mut())
            }),
            [
                Err(ModelError::DecisionTypeNotEnoughMemory),
                Err(ModelError::DecisionTypeLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn reading_the_decision_type_reports_an_exception_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"metadata read threw".as_ptr()) };

        assert_eq!(
            unsafe {
                model_decision_type_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    NOT_A_DECISION_MODEL,
                    out_error,
                )
            },
            Err(ModelError::DecisionTypeReported {
                message: "metadata read threw".to_owned()
            })
        );
    }

    #[test]
    fn reading_the_decision_type_reports_an_exception_without_a_message_as_a_contract_violation() {
        assert_eq!(
            unsafe {
                model_decision_type_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    NOT_A_DECISION_MODEL,
                    ptr::null_mut(),
                )
            },
            Err(ModelError::Ffi(FfiError::ContractViolated {
                operation: "llama_rs_model_decision_type",
                detail: "reported a thrown C++ exception without an error message",
            }))
        );
    }

    #[test]
    fn reading_the_decision_type_preserves_an_unknown_status() {
        assert_eq!(
            unsafe {
                model_decision_type_status_to_result(255, NOT_A_DECISION_MODEL, ptr::null_mut())
            },
            Err(ModelError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_model_decision_type",
                code: 255,
            }))
        );
    }

    #[test]
    fn labels_are_read_in_order_until_the_first_missing_one() {
        let labels = [c"yes", c"no"];

        assert_eq!(
            unsafe { collect_decision_labels(|index| label_at(&labels, index)) },
            Ok(vec!["yes".to_owned(), "no".to_owned()])
        );
    }

    #[test]
    fn a_label_that_is_not_utf8_is_reported_with_its_index() {
        let labels = [c"yes", c"\xff"];

        assert_eq!(
            unsafe { collect_decision_labels(|index| label_at(&labels, index)) },
            Err(ModelError::DecisionLabelNotUtf8 {
                index: 1,
                source: c"\xff".to_str().unwrap_err(),
            })
        );
    }

    #[test]
    fn tokenize_ok_returns_count() {
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_OK,
                12,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(12));
    }

    #[test]
    fn tokenize_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ModelError::TokenizeNotEnoughMemory));
    }

    #[test]
    fn tokenize_cxx_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"tokenizer threw".as_ptr()) };
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            )
        };

        assert_eq!(
            result,
            Err(ModelError::TokenizeReported {
                message: "tokenizer threw".to_owned()
            })
        );
    }

    #[test]
    fn tokenize_cxx_exception_without_a_message_is_a_contract_error() {
        let result = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(FfiError::ContractViolated {
                operation: "llama_rs_tokenize",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn tokenize_unknown_status_is_preserved() {
        let result = unsafe { tokenize_status_to_result(255, 0, ptr::null_mut()) };

        assert_eq!(
            result,
            Err(ModelError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_tokenize",
                code: 255,
            }))
        );
    }

    #[test]
    fn tokenize_status_to_result_reports_a_null_vocab_arg_as_a_contract_violation() {
        let outcome = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_VOCAB_ARG,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null vocab argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_reports_a_null_text_arg_as_a_contract_violation() {
        let outcome = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_TEXT_ARG,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null text argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_reports_a_null_out_returned_count_arg_as_a_contract_violation() {
        let outcome = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_RETURNED_COUNT_ARG,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null out_returned_count argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_reports_a_null_out_error_arg_as_a_contract_violation() {
        let outcome = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_ERROR_ARG,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(outcome.err(), Some(ModelError::TokenizeLlamaCppOutOfMemory));
    }
}
