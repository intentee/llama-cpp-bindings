use log::debug;
use std::ffi::c_char;
use std::ffi::c_void;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::num::NonZeroI32;
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_bindings_sys::bindings::llama_rs_batch_ext_init_status;
use llama_cpp_bindings_sys::bindings::llama_rs_new_context_with_model_status;
use llama_cpp_bindings_sys::bindings::llama_rs_process_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::llama_token_data::LlamaTokenData;
use llama_cpp_model::llama_token_data_array::LlamaTokenDataArray;
use llama_cpp_model::lora_adapter_scale::LoraAdapterScale;
use llama_cpp_params::llama_context_params::LlamaContextParams;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;

use crate::context_error::ContextError;
use crate::context_limits::ContextLimits;
use crate::decoded_output_rows::DecodedOutputRows;
use crate::decoded_outputs::DecodedOutputs;
use crate::llama_batch::LlamaBatch;
use crate::sequence_output_capacity::SequenceOutputCapacity;

const fn check_lora_adapters_result(err_code: i32) -> Result<(), ContextError> {
    if err_code != 0 {
        return Err(ContextError::SetLoraAdaptersFailed(err_code));
    }

    Ok(())
}

/// # Safety
///
/// On `LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK` `out_ctx` must be the non-null context llama.cpp
/// created, and on `LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must
/// be the non-null message the C++ side allocated; the message is freed here.
unsafe fn new_context_with_model_status_to_result(
    status: llama_rs_new_context_with_model_status,
    out_ctx: *mut llama_cpp_bindings_sys::bindings::llama_context,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_context>, ContextError> {
    match status {
        llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_ctx) })
        }
        llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_RETURNED_NULL => {
            Err(ContextError::ContextUnconstructible)
        }
        llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextError::CreateNotEnoughMemory)
        }
        llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextError::CreateLlamaCppOutOfMemory)
        }
        llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_new_context_with_model", |message| {
                    ContextError::CreateReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_BATCH_EXT_INIT_OK` `out_batch` must be the non-null batch llama.cpp allocated, and
/// on `LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null
/// message the C++ side allocated; the message is freed here.
unsafe fn batch_ext_init_status_to_result(
    status: llama_rs_batch_ext_init_status,
    out_batch: *mut llama_cpp_bindings_sys::bindings::llama_batch_ext,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::llama_batch_ext>, ContextError> {
    match status {
        llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_batch) })
        }
        llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextError::CreateNotEnoughMemory)
        }
        llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextError::CreateLlamaCppOutOfMemory)
        }
        llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_batch_ext_init", |message| {
                    ContextError::CreateReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_PROCESS_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message
/// the C++ side allocated; it is freed here.
unsafe fn process_status_to_result(
    status: llama_rs_process_status,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
    return_code_error: fn(NonZeroI32) -> ContextError,
    process_failure: fn(Box<ContextError>) -> ContextError,
) -> Result<(), ContextError> {
    match status {
        llama_rs_process_status::LLAMA_RS_PROCESS_OK => NonZeroI32::new(out_llama_cpp_return_code)
            .map_or(Ok(()), |code| Err(return_code_error(code))),
        llama_rs_process_status::LLAMA_RS_PROCESS_ERROR_STRING_ALLOCATION_FAILED => Err(
            process_failure(Box::new(ContextError::ProcessNotEnoughMemory)),
        ),
        llama_rs_process_status::LLAMA_RS_PROCESS_LLAMA_CPP_OUT_OF_MEMORY => Err(process_failure(
            Box::new(ContextError::ProcessLlamaCppOutOfMemory),
        )),
        llama_rs_process_status::LLAMA_RS_PROCESS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(process_failure(Box::new(
                unsafe { CppAllocatedString::from_raw(out_error) }
                    .into_reported_error("llama_rs_process", |message| {
                        ContextError::ProcessReported { message }
                    }),
            )))
        }
    }
}

fn memory_of(
    context: NonNull<llama_cpp_bindings_sys::bindings::llama_context>,
) -> Option<llama_cpp_bindings_sys::bindings::llama_memory_t> {
    let memory = unsafe { llama_cpp_bindings_sys::bindings::llama_get_memory(context.as_ptr()) };

    (!memory.is_null()).then_some(memory)
}

unsafe fn logits_slice<'logits>(
    logits: NonNull<f32>,
    n_vocab: i32,
) -> Result<&'logits [f32], ContextError> {
    let len = usize::try_from(n_vocab).map_err(ContextError::VocabSizeOverflow)?;

    Ok(unsafe { slice::from_raw_parts(logits.as_ptr(), len) })
}

unsafe extern "C" fn abort_callback_trampoline(data: *mut c_void) -> bool {
    let flag = unsafe { &*(data as *const AtomicBool) };

    flag.load(Ordering::Relaxed)
}

pub struct LlamaContext<'model> {
    context: NonNull<llama_cpp_bindings_sys::bindings::llama_context>,
    model: &'model LlamaModel,
    abort_flag: Option<Arc<AtomicBool>>,
    decoded_outputs: DecodedOutputs,
    batch_ext: NonNull<llama_cpp_bindings_sys::bindings::llama_batch_ext>,
    limits: ContextLimits,
    keeps_no_memory: bool,
    embeddings: bool,
    evaluated_tokens: u64,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    /// # Errors
    ///
    /// Returns [`ContextError`] when llama.cpp fails to allocate the context, or when
    /// the context has more sequences than one batch can output.
    pub fn from_model(
        model: &'model LlamaModel,
        _backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<Self, ContextError> {
        let context_params = params.as_ffi();
        let embeddings = params.embeddings();

        SequenceOutputCapacity {
            attention_type: params.attention_type()?,
            model_causal_attention: unsafe {
                llama_cpp_bindings_sys::bindings::llama_rs_model_causal_attn(model.as_ptr())
            },
            model_has_encoder: model.has_encoder(),
            model_n_ctx_train: unsafe {
                llama_cpp_bindings_sys::bindings::llama_model_n_ctx_train(model.as_ptr())
            }
            .cast_unsigned(),
            n_batch: context_params.n_batch,
            n_ctx: context_params.n_ctx,
            n_outputs_max: context_params.n_outputs_max,
            n_seq_max: context_params.n_seq_max,
        }
        .validate()?;

        let mut out_ctx: *mut llama_cpp_bindings_sys::bindings::llama_context = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_new_context_with_model(
                model.as_ptr(),
                context_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };
        let context =
            unsafe { new_context_with_model_status_to_result(status, out_ctx, out_error) }?;

        Self::with_batch(model, context, embeddings)
    }

    fn with_batch(
        model: &'model LlamaModel,
        context: NonNull<llama_cpp_bindings_sys::bindings::llama_context>,
        embeddings: bool,
    ) -> Result<Self, ContextError> {
        let pooling_type = LlamaPoolingType::try_from(unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_pooling_type(context.as_ptr())
        })
        .inspect_err(|_unknown_pooling_type| unsafe {
            llama_cpp_bindings_sys::bindings::llama_free(context.as_ptr());
        })?;
        let mut out_batch: *mut llama_cpp_bindings_sys::bindings::llama_batch_ext = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_batch_ext_init(
                context.as_ptr(),
                &raw mut out_batch,
                &raw mut out_error,
            )
        };
        let batch_ext = unsafe { batch_ext_init_status_to_result(status, out_batch, out_error) }
            .inspect_err(|_batch_init_failure| unsafe {
                llama_cpp_bindings_sys::bindings::llama_free(context.as_ptr());
            })?;
        let keeps_no_memory = memory_of(context).is_none();

        Ok(Self {
            context,
            model,
            abort_flag: None,
            decoded_outputs: DecodedOutputs {
                rows: DecodedOutputRows::BatchIndices(Vec::new()),
                pooling_type,
            },
            batch_ext,
            limits: ContextLimits {
                batch: unsafe { llama_cpp_bindings_sys::bindings::llama_n_batch(context.as_ptr()) },
                micro_batch: unsafe {
                    llama_cpp_bindings_sys::bindings::llama_n_ubatch(context.as_ptr())
                },
                max_sequences: unsafe {
                    llama_cpp_bindings_sys::bindings::llama_n_seq_max(context.as_ptr())
                },
                vocabulary_size: model.n_vocab(),
            },
            keeps_no_memory,
            embeddings,
            evaluated_tokens: 0,
        })
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut llama_cpp_bindings_sys::bindings::llama_context {
        self.context.as_ptr()
    }

    /// # Errors
    /// Returns `ContextError` if the token is not initialized or out of range.
    pub fn candidates_ith(
        &self,
        token_index: i32,
    ) -> Result<impl Iterator<Item = LlamaTokenData> + '_, ContextError> {
        let logits = self.get_logits_ith(token_index)?;

        Ok((0_i32..).zip(logits).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        }))
    }

    pub fn clear_abort_callback(&mut self) {
        self.abort_flag = None;

        unsafe {
            llama_cpp_bindings_sys::bindings::llama_set_abort_callback(
                self.context.as_ptr(),
                None,
                ptr::null_mut(),
            );
        }
    }

    /// # Errors
    ///
    /// - `ContextError` if the decoding failed.
    pub fn decode(&mut self, batch: &LlamaBatch) -> Result<(), ContextError> {
        if !self.decode_fits_micro_batch(batch.n_tokens().unsigned_abs()) {
            return Err(ContextError::DecodeBatchExceedsMicroBatch {
                n_tokens: batch.n_tokens(),
                n_ubatch: self.n_ubatch(),
            });
        }

        self.transfer_batch(batch)
            .map_err(Box::new)
            .map_err(ContextError::DecodeBatchTransfer)?;
        self.process(
            llama_cpp_bindings_sys::bindings::LLAMA_PROCESS_TYPE_DECODE,
            batch,
            ContextError::from_decode_return_code,
            ContextError::DecodeProcess,
        )
    }

    /// Whether the micro batch of this context admits a decode of `n_tokens` tokens: it admits
    /// every size while the context splits batches into micro batches, and only up to
    /// [`Self::n_ubatch`] tokens while it decodes every batch in a single micro batch, as it does
    /// while the context attends non-causally or keeps no memory. It does not check `n_tokens`
    /// against [`Self::n_batch`], which every decode has to stay within as well.
    #[must_use]
    pub fn decode_fits_micro_batch(&self, n_tokens: u32) -> bool {
        !self.decodes_batches_in_one_micro_batch() || self.fits_micro_batch(n_tokens)
    }

    /// # Errors
    ///
    /// - When embeddings are not enabled on this context.
    /// - When the context pools its outputs, because llama.cpp then writes only sequence
    ///   embeddings and never the token rows.
    /// - When `token_index` names no output row of the last decode. With embeddings enabled,
    ///   llama.cpp outputs every token of the batch.
    pub fn embeddings_ith(&self, token_index: i32) -> Result<&[f32], ContextError> {
        if !self.embeddings_enabled() {
            return Err(ContextError::EmbeddingsNotEnabled);
        }

        let pooling_type = self.pooling_type();

        if pooling_type != LlamaPoolingType::None {
            return Err(ContextError::TokenEmbeddingsRequireNonePooling { pooling_type });
        }

        let embedding = unsafe {
            llama_cpp_bindings_sys::bindings::llama_get_embeddings_ith(
                self.context.as_ptr(),
                token_index,
            )
        };

        if embedding.is_null() {
            Err(ContextError::EmbeddingTokenNotAnOutput { token_index })
        } else {
            self.output_embedding_row(embedding)
        }
    }

    /// # Errors
    ///
    /// - When embeddings are not enabled on this context.
    /// - When this context does not pool its outputs.
    /// - When the given sequence was not part of the last decoded batch.
    pub fn embeddings_seq_ith(&self, sequence_index: i32) -> Result<&[f32], ContextError> {
        if !self.embeddings_enabled() {
            return Err(ContextError::EmbeddingsNotEnabled);
        }

        let sequence_embedding_width = self.sequence_embedding_width()?;
        let embedding = unsafe {
            llama_cpp_bindings_sys::bindings::llama_get_embeddings_seq(
                self.context.as_ptr(),
                sequence_index,
            )
        };

        if embedding.is_null() {
            Err(ContextError::SequenceEmbeddingUnavailable { sequence_index })
        } else {
            Ok(unsafe { slice::from_raw_parts(embedding, sequence_embedding_width) })
        }
    }

    /// # Errors
    ///
    /// - `ContextError` if the encoding failed, or [`ContextError::EncodeModelHasNoEncoder`] when the
    ///   model has no encoder to run.
    pub fn encode(&mut self, batch: &LlamaBatch) -> Result<(), ContextError> {
        if !self.fits_micro_batch(batch.n_tokens().unsigned_abs()) {
            return Err(ContextError::EncodeBatchExceedsMicroBatch {
                n_tokens: batch.n_tokens(),
                n_ubatch: self.n_ubatch(),
            });
        }

        if !self.model.has_encoder() {
            return Err(ContextError::EncodeModelHasNoEncoder);
        }

        self.transfer_batch(batch)
            .map_err(Box::new)
            .map_err(ContextError::EncodeBatchTransfer)?;
        self.process(
            llama_cpp_bindings_sys::bindings::LLAMA_PROCESS_TYPE_ENCODE,
            batch,
            ContextError::from_encode_return_code,
            ContextError::EncodeProcess,
        )
    }

    /// The tokens llama.cpp evaluated on this context: every token of each batch it decoded or
    /// encoded successfully, and every token the mtmd helper decoded.
    #[must_use]
    pub const fn evaluated_tokens(&self) -> u64 {
        self.evaluated_tokens
    }

    /// Replaces the candidates in `token_data_array` with the logits at
    /// `token_index`, reusing its allocation.
    ///
    /// # Errors
    /// Returns `ContextError` if the token is not initialized or out of range.
    pub fn fill_token_data_array_ith(
        &self,
        token_index: i32,
        token_data_array: &mut LlamaTokenDataArray,
    ) -> Result<(), ContextError> {
        token_data_array.replace_candidates(self.candidates_ith(token_index)?);

        Ok(())
    }

    /// Accepts the batch index of an output token, or a negative index counting back from the
    /// last output.
    ///
    /// # Errors
    /// Returns `ContextError` if the token is not an output of the last decode, out of range, the
    /// last decode produced no logits, or `n_vocab` overflows.
    pub fn get_logits_ith(&self, token_index: i32) -> Result<&[f32], ContextError> {
        self.decoded_outputs.ensure_output(token_index)?;
        self.ensure_logits_produced()?;

        let row = NonNull::new(unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_get_logits_ith(
                self.context.as_ptr(),
                token_index,
            )
        })
        .ok_or(ContextError::LogitsNull)?;

        unsafe { logits_slice(row, self.model.n_vocab()) }
    }

    /// The limits a batch must respect to be decoded or encoded on this context.
    #[must_use]
    pub const fn limits(&self) -> ContextLimits {
        self.limits
    }

    /// The memory module of this context, which a context that keeps no memory does not have.
    #[must_use]
    pub fn memory(&self) -> Option<llama_cpp_bindings_sys::bindings::llama_memory_t> {
        memory_of(self.context)
    }

    #[must_use]
    pub const fn model(&self) -> &'model LlamaModel {
        self.model
    }

    #[must_use]
    pub const fn n_batch(&self) -> u32 {
        self.limits.batch
    }

    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::bindings::llama_n_ctx(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_ctx_seq(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::bindings::llama_n_ctx_seq(self.context.as_ptr()) }
    }

    #[must_use]
    pub const fn n_seq_max(&self) -> u32 {
        self.limits.max_sequences
    }

    #[must_use]
    pub const fn n_ubatch(&self) -> u32 {
        self.limits.micro_batch
    }

    #[must_use]
    pub const fn pooling_type(&self) -> LlamaPoolingType {
        self.decoded_outputs.pooling_type
    }

    /// Records the tokens and the rows a llama.cpp helper evaluated and wrote when it decoded on
    /// this context without one of this context's batches, such as the mtmd helper decoding media
    /// chunks.
    pub fn record_helper_decode(&mut self, evaluated_tokens: u64, logits_last: bool) {
        self.evaluated_tokens = self.evaluated_tokens.saturating_add(evaluated_tokens);
        self.decoded_outputs.rows = DecodedOutputRows::of_helper_decode(logits_last);
    }

    pub fn set_abort_flag(&mut self, flag: Arc<AtomicBool>) {
        let raw_ptr = Arc::as_ptr(&flag) as *mut c_void;
        self.abort_flag = Some(flag);

        unsafe {
            llama_cpp_bindings_sys::bindings::llama_set_abort_callback(
                self.context.as_ptr(),
                Some(abort_callback_trampoline),
                raw_ptr,
            );
        }
    }

    pub fn set_embeddings(&mut self, embeddings: bool) {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_set_embeddings(
                self.context.as_ptr(),
                embeddings,
            );
        }
        self.embeddings = embeddings;
    }

    /// # Errors
    ///
    /// Returns [`ContextError::LoraAdapterBelongsToAnotherModel`], before handing anything to
    /// llama.cpp, when an adapter was not loaded into this context's model, or
    /// [`ContextError::SetLoraAdaptersFailed`] when llama.cpp refuses the adapters.
    pub fn set_lora_adapters(&self, adapters: &[LoraAdapterScale<'_>]) -> Result<(), ContextError> {
        let model_adapters = self.model.lora_adapters();

        if let Some(position) = adapters
            .iter()
            .position(|LoraAdapterScale { adapter, .. }| !model_adapters.contains(adapter))
        {
            return Err(ContextError::LoraAdapterBelongsToAnotherModel { position });
        }

        let mut raw_adapters = adapters
            .iter()
            .map(|LoraAdapterScale { adapter, .. }| adapter.as_ptr())
            .collect::<Vec<_>>();
        let mut scales = adapters
            .iter()
            .map(|LoraAdapterScale { scale, .. }| *scale)
            .collect::<Vec<_>>();
        let raw_adapters_ptr = raw_adapters
            .first_mut()
            .map_or(ptr::null_mut(), ptr::from_mut);
        let scales_ptr = scales.first_mut().map_or(ptr::null_mut(), ptr::from_mut);
        let err_code = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_set_adapters_lora(
                self.context.as_ptr(),
                raw_adapters_ptr,
                raw_adapters.len(),
                scales_ptr,
            )
        };
        check_lora_adapters_result(err_code)?;

        debug!("Updated lora adapters");
        Ok(())
    }

    /// # Errors
    /// Returns `ContextError` if the token is not initialized or out of range.
    pub fn token_data_array_ith(
        &self,
        token_index: i32,
    ) -> Result<LlamaTokenDataArray, ContextError> {
        let mut token_data_array = LlamaTokenDataArray::new(Vec::new(), false);

        self.fill_token_data_array_ith(token_index, &mut token_data_array)?;

        Ok(token_data_array)
    }

    fn decodes_batches_in_one_micro_batch(&self) -> bool {
        self.keeps_no_memory
            || !unsafe {
                llama_cpp_bindings_sys::bindings::llama_get_causal_attn(self.context.as_ptr())
            }
    }

    const fn embeddings_enabled(&self) -> bool {
        self.embeddings
    }

    fn ensure_logits_produced(&self) -> Result<(), ContextError> {
        if unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_context_produced_logits(
                self.context.as_ptr(),
            )
        } {
            Ok(())
        } else {
            Err(self.decoded_outputs.missing_logits_error())
        }
    }

    const fn fits_micro_batch(&self, n_tokens: u32) -> bool {
        n_tokens <= self.n_ubatch()
    }

    fn output_embedding_row(&self, embedding: *const f32) -> Result<&[f32], ContextError> {
        self.output_embedding_width()
            .map(|output_embedding_width| unsafe {
                slice::from_raw_parts(embedding, output_embedding_width)
            })
    }

    fn output_embedding_width(&self) -> Result<usize, ContextError> {
        usize::try_from(self.model.n_embd_out()).map_err(ContextError::InvalidEmbeddingDimension)
    }

    fn process(
        &mut self,
        process_type: llama_cpp_bindings_sys::bindings::llama_process_type,
        batch: &LlamaBatch,
        return_code_error: fn(NonZeroI32) -> ContextError,
        process_failure: fn(Box<ContextError>) -> ContextError,
    ) -> Result<(), ContextError> {
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_process(
                self.context.as_ptr(),
                process_type,
                self.batch_ext.as_ptr(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };

        unsafe {
            process_status_to_result(
                status,
                out_llama_cpp_return_code,
                out_error,
                return_code_error,
                process_failure,
            )
        }?;
        self.evaluated_tokens = self
            .evaluated_tokens
            .saturating_add(u64::from(batch.n_tokens().unsigned_abs()));
        self.record_batch_output_rows(batch.output_token_indices());

        Ok(())
    }

    fn record_batch_output_rows(&mut self, output_token_indices: &[i32]) {
        if let DecodedOutputRows::BatchIndices(rows) = &mut self.decoded_outputs.rows {
            rows.clear();
            rows.extend_from_slice(output_token_indices);
        } else {
            self.decoded_outputs.rows =
                DecodedOutputRows::BatchIndices(output_token_indices.to_vec());
        }
    }

    fn sequence_embedding_width(&self) -> Result<usize, ContextError> {
        match self.pooling_type() {
            LlamaPoolingType::Rank => usize::try_from(self.model.n_cls_out())
                .map_err(ContextError::InvalidEmbeddingDimension),
            LlamaPoolingType::Mean | LlamaPoolingType::Cls | LlamaPoolingType::Last => {
                self.output_embedding_width()
            }
            LlamaPoolingType::None | LlamaPoolingType::Unspecified => {
                Err(ContextError::SequenceEmbeddingsRequirePooling)
            }
        }
    }

    fn transfer_batch(&self, batch: &LlamaBatch) -> Result<(), ContextError> {
        if batch.limits() != self.limits {
            return Err(ContextError::BatchLimitsMismatch {
                batch: batch.limits(),
                context: self.limits,
            });
        }

        unsafe { batch.assign_to(self.batch_ext) }
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_batch_ext_free(self.batch_ext.as_ptr());
            llama_cpp_bindings_sys::bindings::llama_free(self.context.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::ptr::NonNull;

    use llama_cpp_bindings_sys::bindings::llama_rs_batch_ext_init_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_new_context_with_model_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_process_status;

    use super::batch_ext_init_status_to_result;
    use super::check_lora_adapters_result;
    use super::logits_slice;
    use super::new_context_with_model_status_to_result;
    use super::process_status_to_result;
    use crate::context_error::ContextError;

    fn decode_process_failure(failure: ContextError) -> ContextError {
        ContextError::DecodeProcess(Box::new(failure))
    }

    #[test]
    fn a_zero_set_adapters_return_code_is_success() {
        assert_eq!(check_lora_adapters_result(0), Ok(()));
    }

    #[test]
    fn a_nonzero_set_adapters_return_code_is_an_error() {
        let result = check_lora_adapters_result(-1);

        assert_eq!(result, Err(ContextError::SetLoraAdaptersFailed(-1)));
    }

    #[test]
    fn new_context_llama_cpp_returned_null_maps_unconstructible() {
        let result = unsafe {
            new_context_with_model_status_to_result(
                llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_RETURNED_NULL,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ContextError::ContextUnconstructible));
    }

    #[test]
    fn new_context_allocation_failed_maps_not_enough_memory() {
        let result = unsafe {
            new_context_with_model_status_to_result(
                llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(ContextError::CreateNotEnoughMemory));
    }

    #[test]
    fn new_context_cxx_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                c"context construction threw".as_ptr(),
            )
        };
        let result = unsafe {
            new_context_with_model_status_to_result(
                llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
        };

        assert_eq!(
            result,
            Err(ContextError::CreateReported {
                message: "context construction threw".to_owned()
            })
        );
    }

    #[test]
    fn new_context_with_model_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            new_context_with_model_status_to_result(
                llama_rs_new_context_with_model_status::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(outcome.err(), Some(ContextError::CreateLlamaCppOutOfMemory));
    }

    #[test]
    fn initializing_a_batch_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED,
                llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| unsafe {
                batch_ext_init_status_to_result(status, ptr::null_mut(), ptr::null_mut())
            }
            .err()),
            [
                Some(ContextError::CreateNotEnoughMemory),
                Some(ContextError::CreateLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn initializing_a_batch_reports_an_exception_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                c"batch allocation threw".as_ptr(),
            )
        };

        assert_eq!(
            unsafe {
                batch_ext_init_status_to_result(
                    llama_rs_batch_ext_init_status::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                )
            }
            .err(),
            Some(ContextError::CreateReported {
                message: "batch allocation threw".to_owned()
            })
        );
    }

    #[test]
    fn every_status_without_a_return_code_or_message_maps_to_its_result() {
        let statuses = [
            llama_rs_process_status::LLAMA_RS_PROCESS_OK,
            llama_rs_process_status::LLAMA_RS_PROCESS_ERROR_STRING_ALLOCATION_FAILED,
            llama_rs_process_status::LLAMA_RS_PROCESS_LLAMA_CPP_OUT_OF_MEMORY,
        ];

        assert_eq!(
            statuses.map(|status| unsafe {
                process_status_to_result(
                    status,
                    0,
                    ptr::null_mut(),
                    ContextError::from_decode_return_code,
                    ContextError::DecodeProcess,
                )
            }),
            [
                Ok(()),
                Err(decode_process_failure(ContextError::ProcessNotEnoughMemory)),
                Err(decode_process_failure(
                    ContextError::ProcessLlamaCppOutOfMemory
                )),
            ]
        );
    }

    #[test]
    fn a_nonzero_llama_cpp_code_is_read_as_the_error_of_the_process_type() {
        assert_eq!(
            [2, -3].map(|code| unsafe {
                process_status_to_result(
                    llama_rs_process_status::LLAMA_RS_PROCESS_OK,
                    code,
                    ptr::null_mut(),
                    ContextError::from_decode_return_code,
                    ContextError::DecodeProcess,
                )
            }),
            [
                Err(ContextError::DecodeAborted),
                Err(ContextError::DecodeComputeFailed)
            ]
        );
        assert_eq!(
            unsafe {
                process_status_to_result(
                    llama_rs_process_status::LLAMA_RS_PROCESS_OK,
                    -2,
                    ptr::null_mut(),
                    ContextError::from_encode_return_code,
                    ContextError::EncodeProcess,
                )
            },
            Err(ContextError::EncodeOutOfMemory)
        );
    }

    #[test]
    fn a_thrown_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"process threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                process_status_to_result(
                    llama_rs_process_status::LLAMA_RS_PROCESS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    0,
                    out_error,
                    ContextError::from_decode_return_code,
                    ContextError::DecodeProcess,
                )
            },
            Err(ContextError::DecodeProcess(Box::new(
                ContextError::ProcessReported {
                    message: "process threw".to_owned()
                }
            )))
        );
    }

    #[test]
    fn a_thrown_exception_whose_message_is_not_utf8_is_a_failure_of_the_process_type() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"\xff".as_ptr()) };

        assert_eq!(
            unsafe {
                process_status_to_result(
                    llama_rs_process_status::LLAMA_RS_PROCESS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    0,
                    out_error,
                    ContextError::from_encode_return_code,
                    ContextError::EncodeProcess,
                )
            },
            Err(ContextError::EncodeProcess(Box::new(
                ContextError::ExceptionMessageNotUtf8 {
                    operation: "llama_rs_process",
                    source: String::from_utf8(vec![0xFF]).unwrap_err(),
                }
            )))
        );
    }

    #[test]
    fn logits_slice_from_negative_vocab_maps_vocab_size_overflow() {
        let logit_value = 0.0_f32;
        let result = unsafe { logits_slice(NonNull::from(&logit_value), -1) };

        let conversion_error = usize::try_from(-1_i32).unwrap_err();

        assert_eq!(
            result,
            Err(ContextError::VocabSizeOverflow(conversion_error))
        );
    }
}
