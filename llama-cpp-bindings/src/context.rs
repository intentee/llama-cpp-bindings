use std::ffi::c_char;
use std::ffi::c_void;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::context::context_limits::ContextLimits;
use crate::context::decoded_output_rows::DecodedOutputRows;
use crate::context::decoded_outputs::DecodedOutputs;
use crate::context::params::LlamaContextParams;
use crate::context::params::LlamaPoolingType;
use crate::context::sequence_output_capacity::SequenceOutputCapacity;
use crate::error::batch_transfer_error::BatchTransferError;
use crate::llama_backend::LlamaBackend;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::model::lora_adapter_scale::LoraAdapterScale;
use crate::timing::LlamaTimings;
use crate::token::LlamaToken;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaContextLoadError, LlamaLoraAdaptersError,
    LogitsError,
};

const fn check_lora_adapters_result(err_code: i32) -> Result<(), LlamaLoraAdaptersError> {
    if err_code != 0 {
        return Err(LlamaLoraAdaptersError::ErrorResult(err_code));
    }

    Ok(())
}

fn nextn_hidden_state_probe_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_context_nextn_hidden_state_probe_status,
    produces_nextn_hidden_state: bool,
    out_error: *mut c_char,
) -> Result<bool, EmbeddingsError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_OK => {
            Ok(produces_nextn_hidden_state)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_PRODUCES_NEXTN_HIDDEN_STATE_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "was given a null out_produces_nextn_hidden_state argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_FAILED_TO_RESERVE_GRAPH => {
            Err(EmbeddingsError::NextnProbeGraphReservationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(EmbeddingsError::NextnProbeNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(EmbeddingsError::NextnProbeLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_context_graph_produces_nextn_hidden_state",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(EmbeddingsError::NextnProbeReported { message })
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_context_graph_produces_nextn_hidden_state",
            code: i64::from(other),
        }
        .into()),
    }
}

fn new_context_with_model_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_new_context_with_model_status,
    out_ctx: *mut llama_cpp_bindings_sys::llama_context,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_context>, LlamaContextLoadError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK => NonNull::new(out_ctx)
            .ok_or_else(|| {
                crate::FfiContractError {
                    operation: "llama_rs_new_context_with_model",
                    detail: "success status contained a null context",
                }
                .into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_RETURNED_NULL => {
            Err(LlamaContextLoadError::Unconstructible)
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED => {
            Err(LlamaContextLoadError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(LlamaContextLoadError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_new_context_with_model",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(LlamaContextLoadError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_MODEL_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_new_context_with_model",
                detail: "was given a null model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_new_context_with_model",
                detail: "was given a null out_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_new_context_with_model",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_new_context_with_model",
            code: i64::from(other),
        }
        .into()),
    }
}

fn batch_ext_init_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_batch_ext_init_status,
    out_batch: *mut llama_cpp_bindings_sys::llama_batch_ext,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_batch_ext>, LlamaContextLoadError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_OK => {
            NonNull::new(out_batch).ok_or_else(|| {
                crate::FfiContractError {
                    operation: "llama_rs_batch_ext_init",
                    detail: "success status contained a null batch",
                }
                .into()
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(LlamaContextLoadError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(LlamaContextLoadError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_batch_ext_init",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(LlamaContextLoadError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_batch_ext_init",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_BATCH_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_batch_ext_init",
                detail: "was given a null out_batch argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_batch_ext_init",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_batch_ext_init",
            code: i64::from(other),
        }
        .into()),
    }
}

fn batch_ext_add_entry_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_batch_ext_add_entry_status,
    out_error: *mut c_char,
) -> Result<(), BatchTransferError> {
    let refused_validated_entry = |detail: &'static str| -> Result<(), BatchTransferError> {
        Err(crate::FfiContractError {
            operation: "llama_rs_batch_ext_add_entry",
            detail,
        }
        .into())
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(BatchTransferError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(BatchTransferError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_batch_ext_add_entry",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(BatchTransferError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_BATCH_FULL => refused_validated_entry(
            "refused an entry of a batch validated against its limits as full",
        ),
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_TOKEN => {
            refused_validated_entry("refused a token validated against the vocabulary")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_SEQUENCE_ID => {
            refused_validated_entry("refused a sequence id validated against the context")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_RETURNED_UNKNOWN_CODE => {
            refused_validated_entry("refused an entry with an undocumented code")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_POSITION_REJECTED => {
            refused_validated_entry("refused the position of an entry it had just added")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_OUTPUT_REJECTED => {
            refused_validated_entry("refused the output flag of an entry it had just added")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_EMPTY_SEQ_IDS => {
            refused_validated_entry("was given an entry without sequence ids")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_BATCH_ARG => {
            refused_validated_entry("was given a null batch argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_SEQ_IDS_ARG => {
            refused_validated_entry("was given a null seq_ids argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG => {
            refused_validated_entry("was given a null out_llama_cpp_return_code argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_ERROR_ARG => {
            refused_validated_entry("was given a null out_error argument")
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_batch_ext_add_entry",
            code: i64::from(other),
        }
        .into()),
    }
}

fn decode_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_decode_status,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), DecodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_RETURNED_NONZERO_CODE => {
            let code =
                NonZeroI32::new(out_llama_cpp_return_code).ok_or(crate::FfiContractError {
                    operation: "llama_rs_decode",
                    detail: "nonzero llama.cpp return status contained zero",
                })?;
            Err(DecodeError::from(code))
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_OUT_OF_MEMORY => {
            Err(DecodeError::DecodeOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_COMPUTE_FAILED => Err(DecodeError::ComputeFailed),
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(DecodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(DecodeError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_decode",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(DecodeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_NULL_CTX_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_decode",
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_NULL_BATCH_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_decode",
            detail: "was given a null batch argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_DECODE_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_decode",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_decode",
            code: i64::from(other),
        }
        .into()),
    }
}

fn encode_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_encode_status,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), EncodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER => {
            Err(EncodeError::ModelHasNoEncoder)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_RETURNED_NONZERO_CODE => {
            let code =
                NonZeroI32::new(out_llama_cpp_return_code).ok_or(crate::FfiContractError {
                    operation: "llama_rs_encode",
                    detail: "nonzero llama.cpp return status contained zero",
                })?;
            Err(EncodeError::from(code))
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_OUT_OF_MEMORY => {
            Err(EncodeError::EncodeOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_COMPUTE_FAILED => Err(EncodeError::ComputeFailed),
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(EncodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(EncodeError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_encode",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(EncodeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_NULL_CTX_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_encode",
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_NULL_BATCH_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_encode",
            detail: "was given a null batch argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_ENCODE_NULL_MODEL => Err(crate::FfiContractError {
            operation: "llama_rs_encode",
            detail: "was given a null model argument",
        }
        .into()),
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_encode",
            code: i64::from(other),
        }
        .into()),
    }
}

unsafe fn logits_slice<'logits>(
    logits: NonNull<f32>,
    n_vocab: i32,
) -> Result<&'logits [f32], LogitsError> {
    let len = usize::try_from(n_vocab).map_err(LogitsError::VocabSizeOverflow)?;

    Ok(unsafe { slice::from_raw_parts(logits.as_ptr(), len) })
}

pub mod context_limits;
pub mod decoded_output_rows;
pub mod decoded_outputs;
pub mod kv_cache;
pub mod kv_cache_type;
pub mod llama_attention_type;
pub mod llama_pooling_type;
pub mod llama_state_seq_flags;
pub mod load_seq_state_error;
pub mod load_session_error;
pub mod loaded_seq_state;
pub mod params;
pub mod rope_scaling_type;
pub mod save_seq_state_error;
pub mod save_session_error;
mod sequence_output_capacity;
pub mod session;
pub mod state_data_error;

unsafe extern "C" fn abort_callback_trampoline(data: *mut c_void) -> bool {
    let flag = unsafe { &*(data as *const AtomicBool) };

    flag.load(Ordering::Relaxed)
}

pub struct LlamaContext<'model> {
    pub context: NonNull<llama_cpp_bindings_sys::llama_context>,
    pub model: &'model LlamaModel,
    abort_flag: Option<Arc<AtomicBool>>,
    decoded_outputs: DecodedOutputs,
    batch_ext: NonNull<llama_cpp_bindings_sys::llama_batch_ext>,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    fn with_batch(
        model: &'model LlamaModel,
        context: NonNull<llama_cpp_bindings_sys::llama_context>,
    ) -> Result<Self, LlamaContextLoadError> {
        let mut out_batch: *mut llama_cpp_bindings_sys::llama_batch_ext = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_batch_ext_init(
                context.as_ptr(),
                &raw mut out_batch,
                &raw mut out_error,
            )
        };
        let batch_ext = batch_ext_init_status_to_result(status, out_batch, out_error).inspect_err(
            |_batch_init_failure| unsafe {
                llama_cpp_bindings_sys::llama_free(context.as_ptr());
            },
        )?;
        let pooling_type = LlamaPoolingType::from(unsafe {
            llama_cpp_bindings_sys::llama_pooling_type(context.as_ptr())
        });

        Ok(Self {
            context,
            model,
            abort_flag: None,
            decoded_outputs: DecodedOutputs {
                rows: DecodedOutputRows::BatchIndices(Vec::new()),
                pooling_type,
            },
            batch_ext,
        })
    }

    /// # Errors
    ///
    /// Returns [`LlamaContextLoadError`] when llama.cpp fails to allocate the context, or when
    /// the context has more sequences than one batch can output.
    pub fn from_model(
        model: &'model LlamaModel,
        _backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<Self, LlamaContextLoadError> {
        let context_params = params.context_params;

        SequenceOutputCapacity {
            attention_type: params.attention_type(),
            model_causal_attention: unsafe {
                llama_cpp_bindings_sys::llama_rs_model_causal_attn(model.model.as_ptr())
            },
            model_has_encoder: unsafe {
                llama_cpp_bindings_sys::llama_model_has_encoder(model.model.as_ptr())
            },
            model_n_ctx_train: unsafe {
                llama_cpp_bindings_sys::llama_model_n_ctx_train(model.model.as_ptr())
            }
            .cast_unsigned(),
            n_batch: context_params.n_batch,
            n_ctx: context_params.n_ctx,
            n_outputs_max: context_params.n_outputs_max,
            n_seq_max: context_params.n_seq_max,
        }
        .validate()?;

        let mut out_ctx: *mut llama_cpp_bindings_sys::llama_context = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_new_context_with_model(
                model.model.as_ptr(),
                context_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };
        let context = new_context_with_model_status_to_result(status, out_ctx, out_error)?;

        Self::with_batch(model, context)
    }

    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_batch(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ubatch(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_seq_max(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_seq_max(self.context.as_ptr()) }
    }

    fn decodes_batches_in_one_micro_batch(&self) -> bool {
        let keeps_no_memory =
            unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) }.is_null();
        let attends_causally =
            unsafe { llama_cpp_bindings_sys::llama_get_causal_attn(self.context.as_ptr()) };

        keeps_no_memory || !attends_causally
    }

    fn exceeds_micro_batch(&self, batch: &LlamaBatch) -> bool {
        i64::from(batch.n_tokens()) > i64::from(self.n_ubatch())
    }

    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ctx(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_ctx_seq(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ctx_seq(self.context.as_ptr()) }
    }

    pub fn set_abort_flag(&mut self, flag: Arc<AtomicBool>) {
        let raw_ptr = Arc::as_ptr(&flag) as *mut c_void;
        self.abort_flag = Some(flag);

        unsafe {
            llama_cpp_bindings_sys::llama_set_abort_callback(
                self.context.as_ptr(),
                Some(abort_callback_trampoline),
                raw_ptr,
            );
        }
    }

    pub fn clear_abort_callback(&mut self) {
        self.abort_flag = None;

        unsafe {
            llama_cpp_bindings_sys::llama_set_abort_callback(
                self.context.as_ptr(),
                None,
                ptr::null_mut(),
            );
        }
    }

    pub fn synchronize(&self) {
        unsafe { llama_cpp_bindings_sys::llama_synchronize(self.context.as_ptr()) }
    }

    pub fn detach_threadpool(&self) {
        unsafe { llama_cpp_bindings_sys::llama_detach_threadpool(self.context.as_ptr()) }
    }

    /// Records which rows the last decode wrote, for a decode llama.cpp ran on this context
    /// without one of this context's batches, such as the mtmd helper decoding media chunks.
    pub fn record_decoded_rows(&mut self, rows: DecodedOutputRows) {
        self.decoded_outputs = DecodedOutputs {
            rows,
            pooling_type: self.pooling_type(),
        };
    }

    fn transfer_batch(&self, batch: &LlamaBatch) -> Result<(), BatchTransferError> {
        let context_limits = ContextLimits::of(self);

        if batch.limits() != context_limits {
            return Err(BatchTransferError::LimitsMismatch {
                batch: batch.limits(),
                context: context_limits,
            });
        }

        unsafe { llama_cpp_bindings_sys::llama_batch_ext_clear(self.batch_ext.as_ptr()) };

        for entry in batch.entries() {
            let mut out_llama_cpp_return_code: i32 = 0;
            let mut out_error: *mut c_char = ptr::null_mut();
            let status = unsafe {
                llama_cpp_bindings_sys::llama_rs_batch_ext_add_entry(
                    self.batch_ext.as_ptr(),
                    entry.token.0,
                    entry.pos,
                    entry.seq_ids.as_ptr(),
                    entry.seq_ids.len(),
                    entry.output,
                    &raw mut out_llama_cpp_return_code,
                    &raw mut out_error,
                )
            };

            batch_ext_add_entry_status_to_result(status, out_error)?;
        }

        Ok(())
    }

    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    pub fn decode(&mut self, batch: &LlamaBatch) -> Result<(), DecodeError> {
        if self.decodes_batches_in_one_micro_batch() && self.exceeds_micro_batch(batch) {
            return Err(DecodeError::BatchExceedsMicroBatch {
                n_tokens: batch.n_tokens(),
                n_ubatch: self.n_ubatch(),
            });
        }

        self.transfer_batch(batch)?;

        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_decode(
                self.context.as_ptr(),
                self.batch_ext.as_ptr(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };
        decode_status_to_result(status, out_llama_cpp_return_code, out_error)?;

        self.record_decoded_rows(DecodedOutputRows::BatchIndices(
            batch.output_token_indices().to_vec(),
        ));

        Ok(())
    }

    /// # Errors
    ///
    /// - `EncodeError` if the encoding failed.
    pub fn encode(&mut self, batch: &LlamaBatch) -> Result<(), EncodeError> {
        if self.exceeds_micro_batch(batch) {
            return Err(EncodeError::BatchExceedsMicroBatch {
                n_tokens: batch.n_tokens(),
                n_ubatch: self.n_ubatch(),
            });
        }

        self.transfer_batch(batch)?;

        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_encode(
                self.context.as_ptr(),
                self.batch_ext.as_ptr(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };
        encode_status_to_result(status, out_llama_cpp_return_code, out_error)?;

        self.record_decoded_rows(DecodedOutputRows::BatchIndices(
            batch.output_token_indices().to_vec(),
        ));

        Ok(())
    }

    #[must_use]
    pub fn pooling_type(&self) -> LlamaPoolingType {
        LlamaPoolingType::from(unsafe {
            llama_cpp_bindings_sys::llama_pooling_type(self.context.as_ptr())
        })
    }

    pub fn set_embeddings(&mut self, embeddings: bool) {
        unsafe {
            llama_cpp_bindings_sys::llama_set_embeddings(self.context.as_ptr(), embeddings);
        }
    }

    fn embeddings_enabled(&self) -> bool {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_context_embeddings_enabled(self.context.as_ptr())
        }
    }

    fn masked_nextn_embeddings_enabled(&self) -> bool {
        let nextn_enabled = unsafe {
            llama_cpp_bindings_sys::llama_rs_context_embeddings_nextn_enabled(self.context.as_ptr())
        };
        let nextn_masked = unsafe {
            llama_cpp_bindings_sys::llama_rs_context_embeddings_nextn_masked(self.context.as_ptr())
        };

        nextn_enabled && nextn_masked
    }

    fn output_embedding_width(&self) -> Result<usize, EmbeddingsError> {
        usize::try_from(self.model.n_embd_out()).map_err(EmbeddingsError::InvalidEmbeddingDimension)
    }

    fn output_embedding_row(&self, embedding: *const f32) -> Result<&[f32], EmbeddingsError> {
        self.output_embedding_width()
            .map(|output_embedding_width| unsafe {
                slice::from_raw_parts(embedding, output_embedding_width)
            })
    }

    fn sequence_embedding_width(&self) -> Result<usize, EmbeddingsError> {
        match self.pooling_type() {
            LlamaPoolingType::Rank => usize::try_from(self.model.n_cls_out())
                .map_err(EmbeddingsError::InvalidEmbeddingDimension),
            LlamaPoolingType::Mean | LlamaPoolingType::Cls | LlamaPoolingType::Last => {
                self.output_embedding_width()
            }
            LlamaPoolingType::None | LlamaPoolingType::Unspecified => {
                Err(EmbeddingsError::NonePoolType)
            }
        }
    }

    /// # Errors
    ///
    /// - When embeddings are not enabled on this context.
    /// - When this context does not pool its outputs.
    /// - When the given sequence was not part of the last decoded batch.
    pub fn embeddings_seq_ith(&self, sequence_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled() {
            return Err(EmbeddingsError::NotEnabled);
        }

        let sequence_embedding_width = self.sequence_embedding_width()?;
        let embedding = unsafe {
            llama_cpp_bindings_sys::llama_get_embeddings_seq(self.context.as_ptr(), sequence_index)
        };

        if embedding.is_null() {
            Err(EmbeddingsError::SequenceEmbeddingUnavailable { sequence_index })
        } else {
            Ok(unsafe { slice::from_raw_parts(embedding, sequence_embedding_width) })
        }
    }

    /// # Errors
    ///
    /// - When embeddings are not enabled on this context.
    /// - When the context pools its outputs, because llama.cpp then writes only sequence
    ///   embeddings and never the token rows.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    pub fn embeddings_ith(&self, token_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled() {
            return Err(EmbeddingsError::NotEnabled);
        }

        let pooling_type = self.pooling_type();

        if pooling_type != LlamaPoolingType::None {
            return Err(EmbeddingsError::TokenEmbeddingsRequireNonePooling { pooling_type });
        }

        let embedding = unsafe {
            llama_cpp_bindings_sys::llama_get_embeddings_ith(self.context.as_ptr(), token_index)
        };

        if embedding.is_null() {
            Err(EmbeddingsError::LogitsNotEnabled)
        } else {
            self.output_embedding_row(embedding)
        }
    }

    /// # Errors
    ///
    /// - When the context pools its outputs, because llama.cpp only extracts `NextN` embeddings
    ///   with `LLAMA_POOLING_TYPE_NONE`.
    /// - When the model's graph produces no `NextN` hidden state; `NextN` embeddings stay disabled.
    /// - When llama.cpp fails while checking the graph; `NextN` embeddings stay disabled.
    pub fn enable_masked_nextn_embeddings(&mut self) -> Result<(), EmbeddingsError> {
        let pooling_type = self.pooling_type();

        if pooling_type != LlamaPoolingType::None {
            return Err(EmbeddingsError::NextnEmbeddingsRequireNonePooling { pooling_type });
        }

        self.set_masked_nextn_embeddings(true);

        let enabled =
            self.graph_produces_nextn_hidden_state()
                .and_then(|produces_nextn_hidden_state| {
                    produces_nextn_hidden_state
                        .then_some(())
                        .ok_or(EmbeddingsError::NextnEmbeddingsUnsupported)
                });

        if enabled.is_err() {
            self.set_masked_nextn_embeddings(false);
        }

        enabled
    }

    fn set_masked_nextn_embeddings(&mut self, enabled: bool) {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_set_embeddings_nextn(
                self.context.as_ptr(),
                enabled,
                enabled,
            );
        }
    }

    fn graph_produces_nextn_hidden_state(&self) -> Result<bool, EmbeddingsError> {
        let mut produces_nextn_hidden_state = false;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_context_graph_produces_nextn_hidden_state(
                self.context.as_ptr(),
                &raw mut produces_nextn_hidden_state,
                &raw mut out_error,
            )
        };

        nextn_hidden_state_probe_status_to_result(status, produces_nextn_hidden_state, out_error)
    }

    /// # Errors
    ///
    /// - When masked `NextN` embeddings were not enabled on this context.
    /// - When the given token was not marked as an output of the last decoded batch.
    pub fn nextn_embeddings_ith(&self, token_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.masked_nextn_embeddings_enabled() {
            return Err(EmbeddingsError::NextnEmbeddingsNotEnabled);
        }

        let embedding = unsafe {
            llama_cpp_bindings_sys::llama_rs_get_embeddings_nextn_ith(
                self.context.as_ptr(),
                token_index,
            )
        };

        if embedding.is_null() {
            Err(EmbeddingsError::NextnEmbeddingUnavailable { token_index })
        } else {
            self.output_embedding_row(embedding)
        }
    }

    /// # Errors
    /// Returns `LogitsError` if logits are null or `n_vocab` overflows.
    pub fn candidates(&self) -> Result<impl Iterator<Item = LlamaTokenData> + '_, LogitsError> {
        let logits = self.get_logits()?;

        Ok((0_i32..).zip(logits).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        }))
    }

    /// # Errors
    /// Returns `LogitsError` if logits are null or `n_vocab` overflows.
    pub fn token_data_array(&self) -> Result<LlamaTokenDataArray, LogitsError> {
        Ok(LlamaTokenDataArray::from_iter(self.candidates()?, false))
    }

    fn produced_logits(&self) -> Result<NonNull<f32>, LogitsError> {
        NonNull::new(unsafe { llama_cpp_bindings_sys::llama_get_logits(self.context.as_ptr()) })
            .ok_or_else(|| self.decoded_outputs.missing_logits_error())
    }

    /// # Errors
    /// Returns `LogitsError` if the last decode produced no logits or `n_vocab` overflows.
    pub fn get_logits(&self) -> Result<&[f32], LogitsError> {
        let logits = self.produced_logits()?;

        unsafe { logits_slice(logits, self.model.n_vocab()) }
    }

    /// # Errors
    /// Returns `LogitsError` if the token is not initialized or out of range.
    pub fn candidates_ith(
        &self,
        token_index: i32,
    ) -> Result<impl Iterator<Item = LlamaTokenData> + '_, LogitsError> {
        let logits = self.get_logits_ith(token_index)?;

        Ok((0_i32..).zip(logits).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        }))
    }

    /// # Errors
    /// Returns `LogitsError` if the token is not initialized or out of range.
    pub fn token_data_array_ith(
        &self,
        token_index: i32,
    ) -> Result<LlamaTokenDataArray, LogitsError> {
        let mut token_data_array = LlamaTokenDataArray::new(Vec::new(), false);

        self.fill_token_data_array_ith(token_index, &mut token_data_array)?;

        Ok(token_data_array)
    }

    /// Replaces the candidates in `token_data_array` with the logits at
    /// `token_index`, reusing its allocation.
    ///
    /// # Errors
    /// Returns `LogitsError` if the token is not initialized or out of range.
    pub fn fill_token_data_array_ith(
        &self,
        token_index: i32,
        token_data_array: &mut LlamaTokenDataArray,
    ) -> Result<(), LogitsError> {
        token_data_array.replace_candidates(self.candidates_ith(token_index)?);

        Ok(())
    }

    /// Accepts the batch index of an output token, or a negative index counting back from the
    /// last output.
    ///
    /// # Errors
    /// Returns `LogitsError` if the token is not an output of the last decode, out of range, the
    /// last decode produced no logits, or `n_vocab` overflows.
    pub fn get_logits_ith(&self, token_index: i32) -> Result<&[f32], LogitsError> {
        self.decoded_outputs.ensure_output(token_index)?;
        self.produced_logits()?;

        let row = NonNull::new(unsafe {
            llama_cpp_bindings_sys::llama_get_logits_ith(self.context.as_ptr(), token_index)
        })
        .ok_or(LogitsError::NullLogits)?;

        unsafe { logits_slice(row, self.model.n_vocab()) }
    }

    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_perf_context_reset(self.context.as_ptr()) }
    }

    pub fn timings(&mut self) -> LlamaTimings {
        LlamaTimings::from(unsafe {
            llama_cpp_bindings_sys::llama_perf_context(self.context.as_ptr())
        })
    }

    /// # Errors
    ///
    /// See [`LlamaLoraAdaptersError`] for more information.
    pub fn set_lora_adapters(
        &self,
        adapters: &[LoraAdapterScale<'_, '_>],
    ) -> Result<(), LlamaLoraAdaptersError> {
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
            llama_cpp_bindings_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                raw_adapters_ptr,
                raw_adapters.len(),
                scales_ptr,
            )
        };
        check_lora_adapters_result(err_code)?;

        log::debug!("Updated lora adapters");
        Ok(())
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_batch_ext_free(self.batch_ext.as_ptr());
            llama_cpp_bindings_sys::llama_free(self.context.as_ptr());
        }
    }
}

#[cfg(test)]
mod check_lora_adapters_result_tests {
    use super::check_lora_adapters_result;
    use crate::LlamaLoraAdaptersError;

    #[test]
    fn check_lora_adapters_result_ok_for_zero() {
        assert!(check_lora_adapters_result(0).is_ok());
    }

    #[test]
    fn check_lora_adapters_result_error_for_nonzero() {
        let result = check_lora_adapters_result(-1);

        assert_eq!(result, Err(LlamaLoraAdaptersError::ErrorResult(-1)));
    }
}

#[cfg(test)]
mod nextn_hidden_state_probe_status_to_result_tests {
    use std::ptr;

    use llama_cpp_ffi_status::read_and_free_cpp_string;
    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer;

    use super::nextn_hidden_state_probe_status_to_result;
    use crate::EmbeddingsError;

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_ok_to_whether_the_graph_produces_it() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_OK,
                true,
                ptr::null_mut(),
            ),
            Ok(true)
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_null_ctx_arg() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_CTX_ARG,
                false,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "was given a null ctx argument",
            }
            .into())
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_null_out_produces_nextn_hidden_state_arg() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_PRODUCES_NEXTN_HIDDEN_STATE_ARG,
                false,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "was given a null out_produces_nextn_hidden_state argument",
            }
            .into())
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_null_out_error_arg() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_ERROR_ARG,
                false,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "was given a null out_error argument",
            }
            .into())
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_llama_cpp_failed_to_reserve_graph() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_FAILED_TO_RESERVE_GRAPH,
                false,
                ptr::null_mut(),
            ),
            Err(EmbeddingsError::NextnProbeGraphReservationFailed)
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_error_string_allocation_failed() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_ERROR_STRING_ALLOCATION_FAILED,
                false,
                ptr::null_mut(),
            ),
            Err(EmbeddingsError::NextnProbeNotEnoughMemory)
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_llama_cpp_out_of_memory() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_OUT_OF_MEMORY,
                false,
                ptr::null_mut(),
            ),
            Err(EmbeddingsError::NextnProbeLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_a_thrown_exception_with_its_message() {
        let expected_message = unsafe {
            read_and_free_cpp_string(
                wrapper_allocated_error_message_pointer(),
                "llama_rs_context_graph_produces_nextn_hidden_state",
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                false,
                wrapper_allocated_error_message_pointer(),
            )
            .err(),
            expected_message
                .ok()
                .map(|message| EmbeddingsError::NextnProbeReported { message })
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_llama_cpp_threw_cxx_exception_without_a_message()
     {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                false,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn nextn_hidden_state_probe_status_to_result_maps_an_unknown_status() {
        assert_eq!(
            nextn_hidden_state_probe_status_to_result(255, false, ptr::null_mut()),
            Err(crate::FfiStatusError {
                operation: "llama_rs_context_graph_produces_nextn_hidden_state",
                code: 255,
            }
            .into())
        );
    }
}

#[cfg(test)]
mod new_context_with_model_status_to_result_tests {
    use std::ptr;

    use super::new_context_with_model_status_to_result;
    use crate::LlamaContextLoadError;

    #[test]
    fn new_context_success_with_null_context_is_contract_error() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(LlamaContextLoadError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_new_context_with_model",
                    detail: "success status contained a null context",
                }
            ))
        );
    }

    #[test]
    fn new_context_llama_cpp_returned_null_maps_unconstructible() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_RETURNED_NULL,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(result, Err(LlamaContextLoadError::Unconstructible));
    }

    #[test]
    fn new_context_allocation_failed_maps_not_enough_memory() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(result, Err(LlamaContextLoadError::NotEnoughMemory));
    }

    #[test]
    fn new_context_cxx_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"context construction threw".as_ptr())
        };
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_THREW_CXX_EXCEPTION,
            ptr::null_mut(),
            out_error,
        );

        assert_eq!(
            result,
            Err(LlamaContextLoadError::Reported {
                message: "context construction threw".to_owned()
            })
        );
    }

    #[test]
    fn new_context_cxx_exception_without_a_message_is_a_contract_error() {
        let result = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_THREW_CXX_EXCEPTION,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "llama_rs_new_context_with_model",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn new_context_unknown_status_is_preserved() {
        let result = new_context_with_model_status_to_result(255, ptr::null_mut(), ptr::null_mut());

        assert_eq!(
            result,
            Err(LlamaContextLoadError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_new_context_with_model",
                code: 255,
            }))
        );
    }

    #[test]
    fn new_context_with_model_status_to_result_maps_null_model_arg() {
        let outcome = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_MODEL_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_new_context_with_model",
                    detail: "was given a null model argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn new_context_with_model_status_to_result_maps_null_out_ctx_arg() {
        let outcome = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_CTX_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_new_context_with_model",
                    detail: "was given a null out_ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn new_context_with_model_status_to_result_maps_null_out_error_arg() {
        let outcome = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_ERROR_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_new_context_with_model",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn new_context_with_model_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = new_context_with_model_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_NEW_CONTEXT_WITH_MODEL_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(LlamaContextLoadError::LlamaCppOutOfMemory)
        );
    }
}

#[cfg(test)]
mod batch_ext_init_status_to_result_tests {
    use std::ptr;

    use super::batch_ext_init_status_to_result;
    use crate::LlamaContextLoadError;

    #[test]
    fn initializing_a_batch_reports_each_null_pointer_as_a_contract_violation() {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_OK,
                "success status contained a null batch",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_NULL_CTX_ARG,
                "was given a null ctx argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_BATCH_ARG,
                "was given a null out_batch argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                batch_ext_init_status_to_result(status, ptr::null_mut(), ptr::null_mut()).err(),
                Some(LlamaContextLoadError::FfiContract(
                    crate::FfiContractError {
                        operation: "llama_rs_batch_ext_init",
                        detail,
                    }
                ))
            );
        }
    }

    #[test]
    fn initializing_a_batch_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| batch_ext_init_status_to_result(
                status,
                ptr::null_mut(),
                ptr::null_mut()
            )
            .err()),
            [
                Some(LlamaContextLoadError::NotEnoughMemory),
                Some(LlamaContextLoadError::LlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn initializing_a_batch_reports_an_exception_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"batch allocation threw".as_ptr())
        };

        assert_eq!(
            batch_ext_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
            .err(),
            Some(LlamaContextLoadError::Reported {
                message: "batch allocation threw".to_owned()
            })
        );
    }

    #[test]
    fn initializing_a_batch_reports_an_exception_without_a_message_as_a_contract_violation() {
        assert_eq!(
            batch_ext_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
            )
            .err(),
            Some(LlamaContextLoadError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_batch_ext_init",
                    detail: "reported a thrown C++ exception without an error message",
                }
            ))
        );
    }

    #[test]
    fn initializing_a_batch_preserves_an_unknown_status() {
        assert_eq!(
            batch_ext_init_status_to_result(255, ptr::null_mut(), ptr::null_mut()).err(),
            Some(LlamaContextLoadError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_batch_ext_init",
                code: 255,
            }))
        );
    }
}

#[cfg(test)]
mod batch_ext_add_entry_status_to_result_tests {
    use std::ptr;

    use super::batch_ext_add_entry_status_to_result;
    use crate::error::batch_transfer_error::BatchTransferError;

    #[test]
    fn adding_a_batch_entry_reports_each_refusal_of_a_validated_entry_as_a_contract_violation() {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_BATCH_FULL,
                "refused an entry of a batch validated against its limits as full",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_TOKEN,
                "refused a token validated against the vocabulary",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_SEQUENCE_ID,
                "refused a sequence id validated against the context",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_RETURNED_UNKNOWN_CODE,
                "refused an entry with an undocumented code",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_POSITION_REJECTED,
                "refused the position of an entry it had just added",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_OUTPUT_REJECTED,
                "refused the output flag of an entry it had just added",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_EMPTY_SEQ_IDS,
                "was given an entry without sequence ids",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_BATCH_ARG,
                "was given a null batch argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_SEQ_IDS_ARG,
                "was given a null seq_ids argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG,
                "was given a null out_llama_cpp_return_code argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                batch_ext_add_entry_status_to_result(status, ptr::null_mut()),
                Err(BatchTransferError::FfiContract(crate::FfiContractError {
                    operation: "llama_rs_batch_ext_add_entry",
                    detail,
                }))
            );
        }
    }

    #[test]
    fn adding_a_batch_entry_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| batch_ext_add_entry_status_to_result(status, ptr::null_mut())),
            [
                Err(BatchTransferError::NotEnoughMemory),
                Err(BatchTransferError::LlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn adding_a_batch_entry_reports_an_exception_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"entry insertion threw".as_ptr())
        };

        assert_eq!(
            batch_ext_add_entry_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                out_error,
            ),
            Err(BatchTransferError::Reported {
                message: "entry insertion threw".to_owned()
            })
        );
    }

    #[test]
    fn adding_a_batch_entry_reports_an_exception_without_a_message_as_a_contract_violation() {
        assert_eq!(
            batch_ext_add_entry_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(BatchTransferError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_batch_ext_add_entry",
                detail: "reported a thrown C++ exception without an error message",
            }))
        );
    }

    #[test]
    fn adding_a_batch_entry_preserves_an_unknown_status() {
        assert_eq!(
            batch_ext_add_entry_status_to_result(255, ptr::null_mut()),
            Err(BatchTransferError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_batch_ext_add_entry",
                code: 255,
            }))
        );
    }
}

#[cfg(test)]
mod decode_status_to_result_tests {
    use std::ptr;

    use super::decode_status_to_result;
    use crate::DecodeError;

    #[test]
    fn decode_nonzero_code_maps_from_code() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_RETURNED_NONZERO_CODE,
            1,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::NoKvCacheSlot));
    }

    #[test]
    fn decode_out_of_memory_maps_decode_out_of_memory() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::DecodeOutOfMemory));
    }

    #[test]
    fn decode_compute_failed_maps_compute_failed() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_COMPUTE_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::ComputeFailed));
    }

    #[test]
    fn decode_allocation_failed_maps_not_enough_memory() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(DecodeError::NotEnoughMemory));
    }

    #[test]
    fn decode_cxx_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"decode threw".as_ptr()) };
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            out_error,
        );

        assert_eq!(
            result,
            Err(DecodeError::Reported {
                message: "decode threw".to_owned()
            })
        );
    }

    #[test]
    fn decode_cxx_exception_without_a_message_is_a_contract_error() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "llama_rs_decode",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn decode_nonzero_status_with_zero_code_is_contract_error() {
        let result = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_RETURNED_NONZERO_CODE,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(DecodeError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_decode",
                detail: "nonzero llama.cpp return status contained zero",
            }))
        );
    }

    #[test]
    fn decode_unknown_status_is_preserved() {
        let result = decode_status_to_result(255, 0, ptr::null_mut());

        assert_eq!(
            result,
            Err(DecodeError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_decode",
                code: 255,
            }))
        );
    }

    #[test]
    fn decode_status_to_result_maps_null_ctx_arg() {
        let outcome = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_decode",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn decode_status_to_result_maps_null_batch_arg() {
        let outcome = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_NULL_BATCH_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_decode",
                    detail: "was given a null batch argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn decode_status_to_result_maps_null_out_error_arg() {
        let outcome = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_NULL_OUT_ERROR_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_decode",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn decode_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = decode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_DECODE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(DecodeError::LlamaCppOutOfMemory));
    }
}

#[cfg(test)]
mod encode_status_to_result_tests {
    use std::ptr;

    use super::encode_status_to_result;
    use crate::EncodeError;

    #[test]
    fn encode_model_has_no_encoder_maps_model_has_no_encoder() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::ModelHasNoEncoder));
    }

    #[test]
    fn encode_nonzero_code_maps_from_code() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_RETURNED_NONZERO_CODE,
            1,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::NoKvCacheSlot));
    }

    #[test]
    fn encode_out_of_memory_maps_encode_out_of_memory() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::EncodeOutOfMemory));
    }

    #[test]
    fn encode_compute_failed_maps_compute_failed() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_COMPUTE_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::ComputeFailed));
    }

    #[test]
    fn encode_allocation_failed_maps_not_enough_memory() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(EncodeError::NotEnoughMemory));
    }

    #[test]
    fn encode_cxx_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"encode threw".as_ptr()) };
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            out_error,
        );

        assert_eq!(
            result,
            Err(EncodeError::Reported {
                message: "encode threw".to_owned()
            })
        );
    }

    #[test]
    fn encode_cxx_exception_without_a_message_is_a_contract_error() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "llama_rs_encode",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn encode_nonzero_status_with_zero_code_is_contract_error() {
        let result = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_RETURNED_NONZERO_CODE,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(EncodeError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_encode",
                detail: "nonzero llama.cpp return status contained zero",
            }))
        );
    }

    #[test]
    fn encode_unknown_status_is_preserved() {
        let result = encode_status_to_result(255, 0, ptr::null_mut());

        assert_eq!(
            result,
            Err(EncodeError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_encode",
                code: 255,
            }))
        );
    }

    #[test]
    fn encode_status_to_result_maps_null_ctx_arg() {
        let outcome = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_encode",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn encode_status_to_result_maps_null_batch_arg() {
        let outcome = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_NULL_BATCH_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_encode",
                    detail: "was given a null batch argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn encode_status_to_result_maps_null_model() {
        let outcome = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_NULL_MODEL,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_encode",
                    detail: "was given a null model argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn encode_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = encode_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_ENCODE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(EncodeError::LlamaCppOutOfMemory));
    }
}

#[cfg(test)]
mod logits_slice_tests {
    use std::ptr::NonNull;

    use super::logits_slice;
    use crate::LogitsError;

    #[test]
    fn logits_slice_from_negative_vocab_maps_vocab_size_overflow() {
        let logit_value = 0.0_f32;
        let result = unsafe { logits_slice(NonNull::from(&logit_value), -1) };

        let conversion_error = usize::try_from(-1_i32).unwrap_err();

        assert_eq!(
            result,
            Err(LogitsError::VocabSizeOverflow(conversion_error))
        );
    }
}
