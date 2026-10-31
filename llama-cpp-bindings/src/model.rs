pub mod add_bos;
pub mod llama_chat_message;
pub mod llama_chat_template;
pub mod llama_lazy_mode;
pub mod llama_lazy_mode_parse_error;
pub mod llama_load_mode;
pub mod llama_load_mode_parse_error;
pub mod llama_lora_adapter;
pub mod llama_split_mode_parse_error;
pub mod lora_adapter_scale;
pub mod params;
pub mod parse_special_tokens;
pub mod rope_type;
pub mod split_mode;
pub mod tokenizer_input;
pub mod vocab_type;
pub mod vocab_type_from_int_error;

use std::ffi::{CStr, CString, c_char};
use std::num::NonZeroU16;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use toktrie::ApproximateTokEnv;
use toktrie::INVALID_TOKEN;
use toktrie::TokRxInfo;
use toktrie::TokTrie;

use crate::chat_templates::ChatTemplates;
use crate::error::chat_template_error::ChatTemplateError;
use crate::error::chat_templates_error::ChatTemplatesError;
use crate::error::llama_lora_adapter_init_error::LlamaLoraAdapterInitError;
use crate::error::llama_model_load_error::LlamaModelLoadError;
use crate::error::meta_val_error::MetaValError;
use crate::error::string_to_token_error::StringToTokenError;
use crate::error::token_attr_error::TokenAttrError;
use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;
use crate::error::token_to_string_error::TokenToStringError;
use crate::llama_backend::LlamaBackend;
use crate::llama_token_attrs::LlamaTokenAttrs;
use crate::model::tokenizer_input::TokenizerInput;
use crate::sampled_token::SampledToken;
use crate::token::LlamaToken;

use crate::model::add_bos::AddBos;
use crate::model::llama_chat_template::LlamaChatTemplate;
use crate::model::llama_lora_adapter::LlamaLoraAdapter;
use crate::model::parse_special_tokens::ParseSpecialTokens;
use crate::model::rope_type::RopeType;
use crate::model::vocab_type::VocabType;
use crate::model::vocab_type_from_int_error::VocabTypeFromIntError;
use llama_cpp_ffi_status::read_and_free_cpp_string;
use params::LlamaModelParams;

const PIECE_PROBE_SIZE: usize = 32;

fn validate_string_length_for_tokenizer(length: usize) -> Result<c_int, StringToTokenError> {
    Ok(c_int::try_from(length)?)
}

fn cstring_with_validated_len(text: &str) -> Result<TokenizerInput, StringToTokenError> {
    let length = validate_string_length_for_tokenizer(text.len())?;
    let c_string = CString::new(text)?;

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

pub struct LlamaModel {
    pub model: NonNull<llama_cpp_bindings_sys::llama_model>,
    tok_env: OnceCell<Arc<ApproximateTokEnv>>,
    chat_templates: OnceCell<ChatTemplates>,
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

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
) -> Result<LlamaModel, LlamaModelLoadError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_OK => {
            let model = NonNull::new(out_model).ok_or_else(|| {
                LlamaModelLoadError::from(llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_load_model_from_file",
                    detail: "success status contained a null model",
                })
            })?;
            Ok(LlamaModel {
                model,
                tok_env: OnceCell::new(),
                chat_templates: OnceCell::new(),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_RETURNED_NULL => {
            if path.exists() {
                Err(LlamaModelLoadError::Unloadable)
            } else {
                Err(LlamaModelLoadError::FileNotFound(path.to_path_buf()))
            }
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(LlamaModelLoadError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(LlamaModelLoadError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_load_model_from_file",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(LlamaModelLoadError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_PATH_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_load_model_from_file",
                detail: "was given a null path argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_MODEL_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_load_model_from_file",
                detail: "was given a null out_model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_ERROR_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_load_model_from_file",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(llama_cpp_ffi_status::FfiStatusError {
            operation: "llama_rs_load_model_from_file",
            code: i64::from(other),
        }
        .into()),
    }
}

impl LlamaModel {
    fn require_vocabulary_token(
        &self,
        token: LlamaToken,
    ) -> Result<LlamaToken, TokenOutsideVocabulary> {
        TokenOutsideVocabulary::check(token, self.n_vocab())
    }

    #[must_use]
    pub fn vocab_ptr(&self) -> *const llama_cpp_bindings_sys::llama_vocab {
        unsafe { llama_cpp_bindings_sys::llama_model_get_vocab(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the value returned by llama.cpp does not fit into a `u32`.
    pub fn n_ctx_train(&self) -> Result<u32, std::num::TryFromIntError> {
        let n_ctx_train =
            unsafe { llama_cpp_bindings_sys::llama_model_n_ctx_train(self.model.as_ptr()) };

        u32::try_from(n_ctx_train)
    }

    pub fn tokens(
        &self,
        decode_special: bool,
    ) -> impl Iterator<Item = (LlamaToken, Result<String, TokenToStringError>)> + '_ {
        (0..self.n_vocab())
            .map(LlamaToken::new)
            .map(move |llama_token| {
                let mut decoder = encoding_rs::UTF_8.new_decoder();
                (
                    llama_token,
                    self.token_to_piece(
                        &SampledToken::Content(llama_token),
                        &mut decoder,
                        decode_special,
                        None,
                    ),
                )
            })
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
    pub fn token_nl(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_nl(self.vocab_ptr()) })
    }

    #[must_use]
    pub fn adds_bos_token(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_vocab_get_add_bos(self.vocab_ptr()) }
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
    pub fn decode_start_token(&self) -> Option<LlamaToken> {
        present_special_token(unsafe {
            llama_cpp_bindings_sys::llama_model_decoder_start_token(self.model.as_ptr())
        })
    }

    #[must_use]
    pub fn token_sep(&self) -> Option<LlamaToken> {
        present_special_token(unsafe { llama_cpp_bindings_sys::llama_vocab_sep(self.vocab_ptr()) })
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
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        let add_bos = match add_bos {
            AddBos::Always => true,
            AddBos::Never => false,
        };
        let parse_special = match parse_special_tokens {
            ParseSpecialTokens::Always => true,
            ParseSpecialTokens::Never => false,
        };

        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + usize::from(add_bos));
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
    /// Returns [`TokenAttrError::TokenOutsideVocabulary`] when the token is not part of this
    /// model's vocabulary, or [`TokenAttrError::UnknownAttributes`] when llama.cpp reports
    /// attributes this library does not know.
    pub fn token_attr(&self, token: LlamaToken) -> Result<LlamaTokenAttrs, TokenAttrError> {
        let LlamaToken(id) = self.require_vocabulary_token(token)?;
        let token_attr =
            unsafe { llama_cpp_bindings_sys::llama_vocab_get_attr(self.vocab_ptr(), id) };

        LlamaTokenAttrs::try_from(token_attr).map_err(TokenAttrError::UnknownAttributes)
    }

    /// # Errors
    ///
    /// - if the token type is unknown
    ///
    /// - if the returned size from llama.cpp does not fit into a `usize`
    pub fn token_to_piece(
        &self,
        token: &SampledToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, TokenToStringError> {
        let (SampledToken::Content(inner)
        | SampledToken::Reasoning(inner)
        | SampledToken::ToolCall(inner)
        | SampledToken::Undeterminable(inner)) = *token;
        let bytes = self.token_to_piece_bytes_fitted(inner, special, lstrip)?;

        let mut output_piece = String::with_capacity(bytes.len());
        let (_result, _decoded_size, _had_replacements) =
            decoder.decode_to_string(&bytes, &mut output_piece, false);

        Ok(output_piece)
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
    ) -> Result<Vec<u8>, TokenToStringError> {
        let token = self.require_vocabulary_token(token)?;
        let buffer_len = c_int::try_from(buffer_size)?;
        let mut buffer: Vec<u8> = vec![0u8; buffer_size];
        let lstrip = lstrip.map_or(0, |strip_count| i32::from(strip_count.get()));
        let size = unsafe {
            llama_cpp_bindings_sys::llama_rs_token_to_piece(
                self.vocab_ptr(),
                token.0,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer_len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
            error_code if error_code.is_negative() => {
                Err(TokenToStringError::InsufficientBufferSpace(error_code))
            }
            size => {
                buffer.truncate(size.unsigned_abs() as usize);

                Ok(buffer)
            }
        }
    }

    fn token_to_piece_bytes_fitted(
        &self,
        token: LlamaToken,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        match self.token_to_piece_bytes(token, PIECE_PROBE_SIZE, special, lstrip) {
            Err(TokenToStringError::InsufficientBufferSpace(required_size)) => self
                .token_to_piece_bytes(
                    token,
                    required_size.unsigned_abs() as usize,
                    special,
                    lstrip,
                ),
            probe_result => probe_result,
        }
    }

    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::llama_vocab_n_tokens(self.vocab_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp emits a vocab type that is not known to this library.
    pub fn vocab_type(&self) -> Result<VocabType, VocabTypeFromIntError> {
        let vocab_type = unsafe { llama_cpp_bindings_sys::llama_vocab_type(self.vocab_ptr()) };

        VocabType::try_from(vocab_type)
    }

    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_bindings_sys::llama_model_n_embd(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn n_embd_out(&self) -> c_int {
        unsafe { llama_cpp_bindings_sys::llama_model_n_embd_out(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn n_cls_out(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_model_n_cls_out(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_size(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_bindings_sys::llama_model_n_params(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the layer count returned by llama.cpp does not fit into a `u32`.
    pub fn n_layer(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_layer(self.model.as_ptr()) })
    }

    /// # Errors
    ///
    /// Returns an error if the head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head(self.model.as_ptr()) })
    }

    /// # Errors
    ///
    /// Returns an error if the KV head count returned by llama.cpp does not fit into a `u32`.
    pub fn n_head_kv(&self) -> Result<u32, std::num::TryFromIntError> {
        u32::try_from(unsafe { llama_cpp_bindings_sys::llama_model_n_head_kv(self.model.as_ptr()) })
    }

    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_is_hybrid(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_model_has_encoder(self.model.as_ptr()) }
    }

    /// # Errors
    /// Returns an error if the key is not found or the value is not valid UTF-8.
    pub fn meta_val_str(&self, key: &str) -> Result<String, MetaValError> {
        let key_cstring = CString::new(key)?;
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

    #[must_use]
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// # Errors
    /// Returns an error if the index is out of range or the key is not valid UTF-8.
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, MetaValError> {
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
    /// Returns an error if the index is out of range or the value is not valid UTF-8.
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, MetaValError> {
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
    pub fn rope_type(&self) -> Option<RopeType> {
        let raw = unsafe { llama_cpp_bindings_sys::llama_model_rope_type(self.model.as_ptr()) };

        RopeType::from_raw(raw)
    }

    /// # Errors
    ///
    /// Returns [`ChatTemplateError::NullError`] when `name` contains a NUL byte, or
    /// [`ChatTemplateError::MissingTemplate`] when the model has no chat template by that name.
    pub fn chat_template(
        &self,
        name: Option<&str>,
    ) -> Result<LlamaChatTemplate, ChatTemplateError> {
        let name_cstr = name.map(CString::new).transpose()?;
        let name_ptr = name_cstr.as_ref().map_or(ptr::null(), |name| name.as_ptr());
        let result = unsafe {
            llama_cpp_bindings_sys::llama_model_chat_template(self.model.as_ptr(), name_ptr)
        };

        if result.is_null() {
            Err(ChatTemplateError::MissingTemplate)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };

            Ok(LlamaChatTemplate(chat_template_cstr.to_owned()))
        }
    }

    /// # Errors
    ///
    /// See [`LlamaModelLoadError`] for more information.
    ///
    /// # Panics
    ///
    /// Panics if a valid UTF-8 path somehow contains interior null bytes (should never happen).
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();

        let path_str = path
            .to_str()
            .ok_or_else(|| LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;
        let cstr = CString::new(path_str)?;

        if !path.exists() {
            return Err(LlamaModelLoadError::FileNotFound(path.to_path_buf()));
        }
        let mut out_model: *mut llama_cpp_bindings_sys::llama_model = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_load_model_from_file(
                cstr.as_ptr(),
                params.params,
                &raw mut out_model,
                &raw mut out_error,
            )
        };
        unsafe { load_model_from_file_status_to_result(status, out_model, out_error, path) }
    }

    /// # Errors
    ///
    /// See [`LlamaLoraAdapterInitError`] for more information.
    pub fn lora_adapter_init(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter<'_>, LlamaLoraAdapterInitError> {
        let path = path.as_ref();

        let path_str = path
            .to_str()
            .ok_or_else(|| LlamaLoraAdapterInitError::PathToStrError(path.to_path_buf()))?;
        let cstr = CString::new(path_str)?;

        if !path.exists() {
            return Err(LlamaLoraAdapterInitError::FileNotFound(path.to_path_buf()));
        }
        initialize_lora_adapter(|| unsafe {
            llama_cpp_bindings_sys::llama_adapter_lora_init(self.model.as_ptr(), cstr.as_ptr())
        })
    }
}

fn initialize_lora_adapter<'model>(
    initialize: impl FnOnce() -> *mut llama_cpp_bindings_sys::llama_adapter_lora,
) -> Result<LlamaLoraAdapter<'model>, LlamaLoraAdapterInitError> {
    let adapter = NonNull::new(initialize()).ok_or(LlamaLoraAdapterInitError::Unloadable)?;
    Ok(LlamaLoraAdapter::new(adapter))
}

impl LlamaModel {
    /// # Errors
    /// Returns [`TokenToStringError`] when a token's byte piece cannot be
    /// retrieved. The legitimate "this token has no byte piece" case is treated
    /// as empty (not an error); a piece that overflows the probe buffer is
    /// re-read at the exact size rather than dropped.
    pub fn approximate_tok_env(&self) -> Result<Arc<ApproximateTokEnv>, TokenToStringError> {
        self.tok_env
            .get_or_try_init(|| build_approximate_tok_env(self))
            .map(Arc::clone)
    }

    /// Returns the model's chat templates, preparing them on first use.
    ///
    /// # Errors
    ///
    /// Returns [`ChatTemplatesError`] when the model has no chat template or llama.cpp cannot
    /// prepare it.
    pub fn chat_templates(&self) -> Result<&ChatTemplates, ChatTemplatesError> {
        self.chat_templates
            .get_or_try_init(|| ChatTemplates::new(self))
    }
}

fn token_piece_bytes_for_tok_env(
    model: &LlamaModel,
    token: LlamaToken,
    special: bool,
) -> Result<Vec<u8>, TokenToStringError> {
    match model.token_to_piece_bytes_fitted(token, special, None) {
        Err(TokenToStringError::UnknownTokenType) => Ok(Vec::new()),
        piece_result => piece_result,
    }
}

fn build_approximate_tok_env(
    model: &LlamaModel,
) -> Result<Arc<ApproximateTokEnv>, TokenToStringError> {
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
        let bytes = token_piece_bytes_for_tok_env(model, token, false)?;
        if bytes.is_empty() {
            let special_bytes = token_piece_bytes_for_tok_env(model, token, true)?;
            if special_bytes.is_empty() {
                words.push(vec![]);
            } else {
                let mut marked = Vec::with_capacity(special_bytes.len() + 1);
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
) -> Result<c_int, StringToTokenError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(StringToTokenError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(StringToTokenError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_tokenize",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(StringToTokenError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_VOCAB_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_tokenize",
                detail: "was given a null vocab argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_TEXT_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_tokenize",
                detail: "was given a null text argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_RETURNED_COUNT_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_tokenize",
                detail: "was given a null out_returned_count argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_NULL_OUT_ERROR_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_tokenize",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(llama_cpp_ffi_status::FfiStatusError {
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
) -> Result<c_int, StringToTokenError> {
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

fn checked_token_buffer_capacity(capacity: usize) -> Result<c_int, StringToTokenError> {
    Ok(c_int::try_from(capacity)?)
}

fn checked_token_count(size: i32) -> Result<usize, StringToTokenError> {
    Ok(usize::try_from(size)?)
}

fn tokenize_into_buffer(
    estimated_capacity: usize,
    invoke: impl Fn(
        *mut llama_cpp_bindings_sys::llama_token,
        c_int,
    ) -> Result<c_int, StringToTokenError>,
) -> Result<Vec<LlamaToken>, StringToTokenError> {
    let buffer_capacity = checked_token_buffer_capacity(estimated_capacity)?;
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
) -> Result<String, MetaValError>
where
    TCFunction: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];
    let result = c_function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len());

    if result < 0 {
        return Err(MetaValError::NegativeReturn(result));
    }

    let returned_len = result.cast_unsigned() as usize;

    if returned_len >= capacity {
        return extract_meta_string(c_function, returned_len + 1);
    }

    if buffer.get(returned_len) != Some(&0) {
        return Err(MetaValError::NegativeReturn(-1));
    }

    buffer.truncate(returned_len);

    Ok(String::from_utf8(buffer)?)
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_model_free(self.model.as_ptr()) }
    }
}

#[cfg(test)]
mod cstring_with_validated_len_tests {
    use std::ffi::CString;
    use std::str;

    use super::cstring_with_validated_len;
    use crate::error::string_to_token_error::StringToTokenError;

    #[test]
    fn cstring_with_validated_len_null_byte_returns_error() {
        assert_eq!(
            cstring_with_validated_len("null\0byte").err(),
            CString::new("null\0byte")
                .err()
                .map(StringToTokenError::NulError)
        );
    }

    #[test]
    fn a_text_longer_than_a_c_int_is_refused_before_it_is_copied() {
        let too_long = vec![0_u8; usize::try_from(i32::MAX).unwrap() + 1];

        assert_eq!(
            cstring_with_validated_len(str::from_utf8(&too_long).unwrap()).err(),
            i32::try_from(too_long.len())
                .err()
                .map(StringToTokenError::CIntConversionError)
        );
    }
}

#[cfg(test)]
mod validate_string_length_for_tokenizer_tests {
    use super::validate_string_length_for_tokenizer;

    #[test]
    fn validate_string_length_overflow_returns_error() {
        assert_eq!(
            validate_string_length_for_tokenizer(usize::MAX),
            i32::try_from(usize::MAX).map_err(
                crate::error::string_to_token_error::StringToTokenError::CIntConversionError
            )
        );
    }
}

#[cfg(test)]
mod checked_token_buffer_capacity_tests {
    use super::checked_token_buffer_capacity;

    #[test]
    fn checked_token_buffer_capacity_overflow_returns_error() {
        assert!(checked_token_buffer_capacity(usize::MAX).is_err());
    }

    #[test]
    fn checked_token_buffer_capacity_in_range_returns_value() {
        assert_eq!(checked_token_buffer_capacity(8), Ok(8));
    }
}

#[cfg(test)]
mod checked_token_count_tests {
    use super::checked_token_count;

    #[test]
    fn checked_token_count_negative_returns_error() {
        assert!(checked_token_count(-1).is_err());
    }

    #[test]
    fn checked_token_count_non_negative_returns_value() {
        assert_eq!(checked_token_count(5), Ok(5));
    }
}

#[cfg(test)]
mod tokenize_into_buffer_tests {

    use super::tokenize_into_buffer;

    #[test]
    fn tokenize_into_buffer_single_pass_sets_length() {
        let buffer = tokenize_into_buffer(8, |_tokens, _n_tokens_max| Ok(3)).unwrap();

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn tokenize_into_buffer_grows_buffer_when_first_pass_reports_negative_size() {
        let call_count = std::cell::Cell::new(0);
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
            Err(crate::error::string_to_token_error::StringToTokenError::NotEnoughMemory)
        });

        assert_eq!(
            result,
            Err(crate::error::string_to_token_error::StringToTokenError::NotEnoughMemory)
        );
    }

    #[test]
    fn tokenize_into_buffer_propagates_second_invocation_error() {
        let call_count = std::cell::Cell::new(0);
        let result = tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 {
                Ok(-20)
            } else {
                Err(crate::error::string_to_token_error::StringToTokenError::NotEnoughMemory)
            }
        });

        assert_eq!(
            result,
            Err(crate::error::string_to_token_error::StringToTokenError::NotEnoughMemory)
        );
        assert_eq!(call_count.get(), 2);
    }

    #[test]
    fn tokenize_into_buffer_refuses_a_capacity_beyond_a_c_int_before_allocating() {
        assert_eq!(
            tokenize_into_buffer(usize::MAX, |_tokens, _n_tokens_max| Ok(0)),
            Err(
                crate::error::string_to_token_error::StringToTokenError::CIntConversionError(
                    i32::try_from(usize::MAX).unwrap_err()
                )
            )
        );
    }

    #[test]
    fn tokenize_into_buffer_refuses_a_required_size_it_cannot_negate() {
        assert_eq!(
            tokenize_into_buffer(8, |_tokens, _n_tokens_max| Ok(i32::MIN)),
            Err(
                crate::error::string_to_token_error::StringToTokenError::CIntConversionError(
                    usize::try_from(i32::MIN).unwrap_err()
                )
            )
        );
    }

    #[test]
    fn tokenize_into_buffer_negative_final_size_returns_conversion_error() {
        let call_count = std::cell::Cell::new(0);
        let result = tokenize_into_buffer(8, |_tokens, _n_tokens_max| {
            let count = call_count.get();
            call_count.set(count + 1);
            if count == 0 { Ok(-20) } else { Ok(-5) }
        });

        assert_eq!(
            result.unwrap_err(),
            crate::error::string_to_token_error::StringToTokenError::CIntConversionError(
                usize::try_from(-5i32).unwrap_err()
            )
        );
    }
}

#[cfg(test)]
mod extract_meta_string_tests {
    use super::extract_meta_string;
    use crate::error::meta_val_error::MetaValError;

    #[test]
    fn returns_error_when_null_terminator_missing() {
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                buffer[0] = b'a';
                buffer[1] = b'b';
                buffer[2] = b'c';
                2
            },
            4,
        );

        assert_eq!(result.unwrap_err(), MetaValError::NegativeReturn(-1));
    }

    #[test]
    fn returns_error_for_negative_return_value() {
        let result = extract_meta_string(|_buf_ptr, _buf_len| -5, 4);

        assert_eq!(result.unwrap_err(), MetaValError::NegativeReturn(-5));
    }

    #[test]
    fn returns_error_for_invalid_utf8_data() {
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                buffer[0] = 0xFF;
                buffer[1] = 0xFE;
                buffer[2] = 0;
                2
            },
            4,
        );

        let invalid_utf8 = String::from_utf8(vec![0xFF, 0xFE]).unwrap_err();

        assert_eq!(result, Err(MetaValError::FromUtf8Error(invalid_utf8)));
    }

    #[test]
    fn triggers_buffer_resize_when_returned_len_exceeds_capacity() {
        let initial_capacity: usize = 4;
        let length_exceeding_initial_capacity = 10;
        let written_length = 2;
        let call_count = std::cell::Cell::new(0);
        let result = extract_meta_string(
            |buf_ptr, buf_len| {
                let count = call_count.get();
                call_count.set(count + 1);
                if count == 0 {
                    length_exceeding_initial_capacity
                } else {
                    let buffer =
                        unsafe { std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), buf_len) };
                    buffer[0] = b'h';
                    buffer[1] = b'i';
                    buffer[2] = 0;
                    written_length
                }
            },
            initial_capacity,
        );

        assert_eq!(result.unwrap(), "hi");
    }
}

#[cfg(test)]
mod initialize_lora_adapter_tests {
    use std::ptr;

    use super::initialize_lora_adapter;
    use crate::error::llama_lora_adapter_init_error::LlamaLoraAdapterInitError;

    #[test]
    fn lora_adapter_initialization_maps_a_null_pointer_to_unloadable() {
        let result = initialize_lora_adapter(ptr::null_mut);

        assert_eq!(result.unwrap_err(), LlamaLoraAdapterInitError::Unloadable);
    }

    #[test]
    fn lora_adapter_initialization_owns_a_valid_pointer() {
        let pointer = ptr::NonNull::dangling();
        let adapter = std::mem::ManuallyDrop::new(
            initialize_lora_adapter(|| pointer.as_ptr())
                .expect("a non-null adapter pointer must initialize"),
        );

        assert_eq!(adapter.as_ptr(), pointer.as_ptr());
    }
}

#[cfg(test)]
mod load_model_from_file_status_to_result_tests {
    use std::path::Path;
    use std::ptr;

    use super::load_model_from_file_status_to_result;
    use crate::error::llama_model_load_error::LlamaModelLoadError;

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
            LlamaModelLoadError::FfiContract(llama_cpp_ffi_status::FfiContractError {
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
            LlamaModelLoadError::FileNotFound(
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

        assert_eq!(result.unwrap_err(), LlamaModelLoadError::NotEnoughMemory);
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
            Some(LlamaModelLoadError::Reported {
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
            llama_cpp_ffi_status::FfiContractError {
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
            LlamaModelLoadError::FfiStatus(llama_cpp_ffi_status::FfiStatusError {
                operation: "llama_rs_load_model_from_file",
                code: 255,
            })
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_maps_null_path_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_load_model_from_file",
                    detail: "was given a null path argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_maps_null_out_model_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_load_model_from_file",
                    detail: "was given a null out_model argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_maps_null_out_error_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_load_model_from_file",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn load_model_from_file_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = unsafe {
            load_model_from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_LOAD_MODEL_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
                Path::new("/missing-for-contract-test.gguf"),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(LlamaModelLoadError::LlamaCppOutOfMemory)
        );
    }
}

#[cfg(test)]
mod tokenize_status_to_result_tests {
    use std::ptr;

    use super::tokenize_status_to_result;
    use crate::error::string_to_token_error::StringToTokenError;

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

        assert_eq!(result, Err(StringToTokenError::NotEnoughMemory));
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
            Err(StringToTokenError::Reported {
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
            Err(llama_cpp_ffi_status::FfiContractError {
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
            Err(StringToTokenError::FfiStatus(
                llama_cpp_ffi_status::FfiStatusError {
                    operation: "llama_rs_tokenize",
                    code: 255,
                }
            ))
        );
    }

    #[test]
    fn tokenize_status_to_result_maps_null_vocab_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null vocab argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_maps_null_text_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null text argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_maps_null_out_returned_count_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null out_returned_count argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_maps_null_out_error_arg() {
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
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_tokenize",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn tokenize_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = unsafe {
            tokenize_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(outcome.err(), Some(StringToTokenError::LlamaCppOutOfMemory));
    }
}
