use std::borrow::Borrow;
use std::ffi::{CString, c_char};
use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

use llama_cpp_error_recorder::ErrorScope;
use llama_cpp_error_recorder::RecordedError;

use crate::context::LlamaContext;
use crate::dry_sequence_breakers::dry_sequence_breakers;
use crate::model::LlamaModel;
use crate::sanitized_grammar::SanitizedGrammar;
use crate::token::LlamaToken;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::logit_bias::LlamaLogitBias;
use crate::{GrammarError, SampleError, SamplerAcceptError, SamplingError};
use llama_cpp_ffi_status::read_and_free_cpp_string;
use llama_cpp_gbnf::gbnf_validation_error::GbnfValidationError;
use llama_cpp_gbnf::validate_gbnf::validate_gbnf;

fn check_sampler_accept_status(
    status: llama_cpp_bindings_sys::llama_rs_sampler_accept_status,
    error_ptr: *mut c_char,
) -> Result<(), SamplerAcceptError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SamplerAcceptError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(SamplerAcceptError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_sampler_accept",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(SamplerAcceptError::GrammarStateCorrupted { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_accept",
                detail: "was given a null sampler argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_accept",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_accept",
            code: i64::from(other),
        }
        .into()),
    }
}

fn sampler_sample_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_sample_status,
    token: i32,
    error_ptr: *mut c_char,
) -> Result<LlamaToken, SampleError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_OK => Ok(LlamaToken(token)),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SampleError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(SampleError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_sampler_sample",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(SampleError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_SAMPLER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_sample",
                detail: "was given a null sampler argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_sample",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_TOKEN_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_sample",
                detail: "was given a null out_token argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_sample",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_sample",
            code: i64::from(other),
        }
        .into()),
    }
}

fn sampler_init_grammar_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_status,
    sampler: *mut llama_cpp_bindings_sys::llama_sampler,
    error_ptr: *mut c_char,
) -> Result<LlamaSampler, GrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_OK => {
            LlamaSampler::from_raw(sampler, "grammar").map_err(Into::into)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_RETURNED_NULL => {
            Err(GrammarError::GrammarMalformed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GrammarError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_sampler_init_grammar",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(GrammarError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_SAMPLER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_init_grammar",
                detail: "was given a null out_sampler argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_init_grammar",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_init_grammar",
            code: i64::from(other),
        }
        .into()),
    }
}

fn sampler_init_grammar_lazy_patterns_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_lazy_patterns_status,
    sampler: *mut llama_cpp_bindings_sys::llama_sampler,
    error_ptr: *mut c_char,
) -> Result<LlamaSampler, GrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_OK => {
            LlamaSampler::from_raw(sampler, "lazy grammar").map_err(Into::into)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_RETURNED_NULL => {
            Err(GrammarError::LazyGrammarMalformed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GrammarError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_INVALID_TRIGGER_PATTERN => {
            let message = unsafe { read_and_free_cpp_string(error_ptr, "llama_rs_sampler_init_grammar_lazy_patterns", "reported a thrown C++ exception without an error message") }?;
            Err(GrammarError::InvalidTriggerPattern { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_string(error_ptr, "llama_rs_sampler_init_grammar_lazy_patterns", "reported a thrown C++ exception without an error message") }?;
            Err(GrammarError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_SAMPLER_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_sampler_init_grammar_lazy_patterns",
            detail: "was given a null out_sampler argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_ERROR_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_sampler_init_grammar_lazy_patterns",
            detail: "was given a null out_error argument",
        }
        .into()),
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_init_grammar_lazy_patterns",
            code: i64::from(other),
        }
        .into()),
    }
}

fn sampler_init_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_init_status,
    sampler: *mut llama_cpp_bindings_sys::llama_sampler,
    error_ptr: *mut c_char,
    sampler_name: &'static str,
) -> Result<LlamaSampler, SamplingError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_OK => {
            LlamaSampler::from_raw(sampler, sampler_name)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_RETURNED_NULL => {
            Err(SamplingError::SamplerUnavailable {
                sampler: sampler_name,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SamplingError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(SamplingError::LlamaCppOutOfMemory {
                sampler: sampler_name,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_sampler_init",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(SamplingError::Reported {
                sampler: sampler_name,
                message,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_NULL_OUT_SAMPLER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_init",
                detail: "was given a null out_sampler argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_init",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_init",
            code: i64::from(other),
        }
        .into()),
    }
}

fn sampler_chain_add_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_chain_add_status,
    error_ptr: *mut c_char,
) -> Result<(), SamplingError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SamplingError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(SamplingError::LlamaCppOutOfMemory { sampler: "chain" })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_sampler_chain_add",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(SamplingError::Reported {
                sampler: "chain",
                message,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_CHAIN_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "was given a null chain argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_SAMPLER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "was given a null sampler argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_chain_add",
            code: i64::from(other),
        }
        .into()),
    }
}

fn checked_usize_as_i32_sampling(value: usize) -> Result<i32, SamplingError> {
    i32::try_from(value).map_err(SamplingError::IntegerOverflow)
}

pub struct LlamaSampler {
    sampler: NonNull<llama_cpp_bindings_sys::llama_sampler>,
}

unsafe impl Send for LlamaSampler {}

fn grammar_callback_error_to_result(error: Option<RecordedError>) -> Result<(), SampleError> {
    error.map_or(Ok(()), |recorded| {
        Err(SampleError::GrammarCallbackFailed {
            message: recorded.into_message(),
        })
    })
}

fn grammar_callback_error_to_accept_result(
    error: Option<RecordedError>,
) -> Result<(), SamplerAcceptError> {
    error.map_or(Ok(()), |recorded| {
        Err(SamplerAcceptError::GrammarCallbackFailed {
            message: recorded.into_message(),
        })
    })
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSampler").finish_non_exhaustive()
    }
}

impl LlamaSampler {
    pub(crate) fn from_raw(
        sampler: *mut llama_cpp_bindings_sys::llama_sampler,
        sampler_name: &'static str,
    ) -> Result<Self, SamplingError> {
        NonNull::new(sampler).map(|sampler| Self { sampler }).ok_or(
            SamplingError::SamplerUnavailable {
                sampler: sampler_name,
            },
        )
    }

    /// Initializes a sampler through one of the `llama_rs_sampler_init*` wrappers, which
    /// report C++ exceptions thrown by llama.cpp instead of letting them cross into Rust.
    ///
    /// # Errors
    ///
    /// Returns [`SamplingError`] if llama.cpp cannot initialize the sampler.
    pub fn initialize(
        sampler_name: &'static str,
        initialize_sampler: impl FnOnce(
            *mut *mut llama_cpp_bindings_sys::llama_sampler,
            *mut *mut c_char,
        ) -> llama_cpp_bindings_sys::llama_rs_sampler_init_status,
    ) -> Result<Self, SamplingError> {
        let mut sampler: *mut llama_cpp_bindings_sys::llama_sampler = std::ptr::null_mut();
        let mut error_ptr: *mut c_char = std::ptr::null_mut();
        let status = initialize_sampler(&raw mut sampler, &raw mut error_ptr);

        sampler_init_status_to_result(status, sampler, error_ptr, sampler_name)
    }

    fn take_into_chain(&self, sampler: Self) -> Result<(), SamplingError> {
        let mut error_ptr: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_chain_add(
                self.sampler.as_ptr(),
                sampler.sampler.as_ptr(),
                &raw mut error_ptr,
            )
        };

        sampler_chain_add_status_to_result(status, error_ptr).map(|()| std::mem::forget(sampler))
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut llama_cpp_bindings_sys::llama_sampler {
        self.sampler.as_ptr()
    }

    /// # Errors
    ///
    /// Returns [`SampleError::Logits`] when the context holds no logits for `idx`, or another
    /// [`SampleError`] if the C++ sampler throws an exception or the grammar sampler callback
    /// recorded a failure during sampling.
    pub fn sample(&mut self, ctx: &LlamaContext, idx: i32) -> Result<LlamaToken, SampleError> {
        ctx.get_logits_ith(idx)?;

        let mut token: i32 = -1;
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let scope = ErrorScope::enter();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_sample(
                self.sampler.as_ptr(),
                ctx.context.as_ptr(),
                idx,
                &raw mut token,
                &raw mut error_ptr,
            )
        };
        let sampled = sampler_sample_status_to_result(status, token, error_ptr);
        grammar_callback_error_to_result(scope.take())?;

        sampled
    }

    /// # Errors
    ///
    /// Returns [`SampleError::GrammarCallbackFailed`] if the grammar sampler callback recorded a
    /// failure during application, which takes precedence over any other failure, or
    /// [`SampleError::SamplerApply`] if applying the sampler fails, including when it excludes
    /// every candidate.
    pub fn apply(&self, data_array: &mut LlamaTokenDataArray) -> Result<(), SampleError> {
        let scope = ErrorScope::enter();
        let applied = data_array.apply_sampler(self);
        grammar_callback_error_to_result(scope.take())?;

        Ok(applied?)
    }

    /// # Errors
    /// Returns [`SamplerAcceptError`] if the underlying sampler rejects the token.
    pub fn accept(&mut self, token: LlamaToken) -> Result<(), SamplerAcceptError> {
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let scope = ErrorScope::enter();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_accept(
                self.sampler.as_ptr(),
                token.0,
                &raw mut error_ptr,
            )
        };
        let accepted = check_sampler_accept_status(status, error_ptr);
        grammar_callback_error_to_accept_result(scope.take())?;

        accepted
    }

    /// # Errors
    /// Returns [`SamplerAcceptError`] if the underlying sampler rejects any token.
    pub fn accept_many(
        &mut self,
        tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>,
    ) -> Result<(), SamplerAcceptError> {
        for token in tokens {
            self.accept(*token.borrow())?;
        }

        Ok(())
    }

    /// # Errors
    /// Returns [`SamplerAcceptError`] if the underlying sampler rejects any token.
    pub fn with_tokens(
        mut self,
        tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>,
    ) -> Result<Self, SamplerAcceptError> {
        self.accept_many(tokens)?;

        Ok(self)
    }

    /// # Errors
    ///
    /// Returns [`SampleError`] if the grammar sampler callback recorded a failure during reset.
    pub fn reset(&mut self) -> Result<(), SampleError> {
        let scope = ErrorScope::enter();
        unsafe {
            llama_cpp_bindings_sys::llama_sampler_reset(self.sampler.as_ptr());
        }

        grammar_callback_error_to_result(scope.take())
    }

    #[must_use]
    pub fn get_seed(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_sampler_get_seed(self.sampler.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the sampler chain cannot be initialized.
    pub fn chain(
        samplers: impl IntoIterator<Item = Self>,
        no_perf: bool,
    ) -> Result<Self, SamplingError> {
        Self::initialize("chain", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_chain_init(
                llama_cpp_bindings_sys::llama_sampler_chain_params { no_perf },
                out_sampler,
                out_error,
            )
        })
        .and_then(|chain| {
            samplers
                .into_iter()
                .try_for_each(|sampler| chain.take_into_chain(sampler))
                .map(|()| chain)
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the sampler chain cannot be initialized.
    pub fn chain_simple(samplers: impl IntoIterator<Item = Self>) -> Result<Self, SamplingError> {
        Self::chain(samplers, false)
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the temperature sampler cannot be initialized.
    pub fn temp(temperature: f32) -> Result<Self, SamplingError> {
        Self::initialize("temperature", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_temp(temperature, out_sampler, out_error)
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the extended temperature sampler cannot be initialized.
    pub fn temp_ext(temperature: f32, delta: f32, exponent: f32) -> Result<Self, SamplingError> {
        Self::initialize("extended temperature", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_temp_ext(
                temperature,
                delta,
                exponent,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the top-k sampler cannot be initialized.
    pub fn top_k(keep_count: i32) -> Result<Self, SamplingError> {
        Self::initialize("top-k", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_top_k(keep_count, out_sampler, out_error)
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the top-n-sigma sampler cannot be initialized.
    pub fn top_n_sigma(sigma_multiplier: f32) -> Result<Self, SamplingError> {
        Self::initialize("top-n-sigma", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_top_n_sigma(
                sigma_multiplier,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the typical sampler cannot be initialized.
    pub fn typical(probability: f32, min_keep: usize) -> Result<Self, SamplingError> {
        Self::initialize("typical", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_typical(
                probability,
                min_keep,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the top-p sampler cannot be initialized.
    pub fn top_p(probability: f32, min_keep: usize) -> Result<Self, SamplingError> {
        Self::initialize("top-p", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_top_p(
                probability,
                min_keep,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the min-p sampler cannot be initialized.
    pub fn min_p(probability: f32, min_keep: usize) -> Result<Self, SamplingError> {
        Self::initialize("min-p", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_min_p(
                probability,
                min_keep,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the XTC sampler cannot be initialized.
    pub fn xtc(
        probability: f32,
        temperature: f32,
        min_keep: usize,
        seed: u32,
    ) -> Result<Self, SamplingError> {
        Self::initialize("XTC", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_xtc(
                probability,
                temperature,
                min_keep,
                seed,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    /// Returns an error if the grammar is invalid or the sampler cannot be initialized.
    pub fn grammar(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<Self, GrammarError> {
        let SanitizedGrammar {
            grammar: grammar_cstring,
            root: root_cstring,
        } = Self::sanitize_grammar_strings(grammar_str, grammar_root)?;
        let mut sampler: *mut llama_cpp_bindings_sys::llama_sampler = std::ptr::null_mut();
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_grammar(
                model.vocab_ptr(),
                grammar_cstring.as_ptr(),
                root_cstring.as_ptr(),
                &raw mut sampler,
                &raw mut error_ptr,
            )
        };

        sampler_init_grammar_status_to_result(status, sampler, error_ptr).map_err(|init_error| {
            Self::diagnose_rejected_grammar(model, grammar_str, grammar_root, init_error)
        })
    }

    /// # Errors
    /// Returns an error if the grammar or trigger patterns are invalid.
    pub fn grammar_lazy(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
        trigger_patterns: &[String],
        trigger_tokens: &[LlamaToken],
    ) -> Result<Self, GrammarError> {
        let SanitizedGrammar {
            grammar: grammar_cstring,
            root: root_cstring,
        } = Self::sanitize_grammar_strings(grammar_str, grammar_root)?;
        let trigger_patterns = Self::sanitize_trigger_patterns(trigger_patterns)?;
        let mut sampler: *mut llama_cpp_bindings_sys::llama_sampler = std::ptr::null_mut();
        let mut error_ptr: *mut c_char = std::ptr::null_mut();

        let mut trigger_pattern_ptrs: Vec<*const c_char> =
            trigger_patterns.iter().map(|cs| cs.as_ptr()).collect();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_grammar_lazy_patterns(
                model.vocab_ptr(),
                grammar_cstring.as_ptr(),
                root_cstring.as_ptr(),
                trigger_pattern_ptrs.as_mut_ptr(),
                trigger_pattern_ptrs.len(),
                trigger_tokens.as_ptr().cast(),
                trigger_tokens.len(),
                &raw mut sampler,
                &raw mut error_ptr,
            )
        };

        sampler_init_grammar_lazy_patterns_status_to_result(status, sampler, error_ptr).map_err(
            |init_error| {
                Self::diagnose_rejected_grammar(model, grammar_str, grammar_root, init_error)
            },
        )
    }

    /// # Errors
    ///
    /// Returns [`GrammarError`] if the grammar is invalid or the sampler cannot be initialized.
    pub fn llguidance(
        model: &LlamaModel,
        grammar_kind: &str,
        grammar_data: &str,
    ) -> Result<Self, GrammarError> {
        crate::llguidance_sampler::create_llg_sampler(model, grammar_kind, grammar_data)
    }

    fn sanitize_grammar_strings(
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<SanitizedGrammar, GrammarError> {
        Ok(SanitizedGrammar {
            grammar: CString::new(grammar_str).map_err(GrammarError::GrammarContainsNul)?,
            root: CString::new(grammar_root).map_err(GrammarError::RootContainsNul)?,
        })
    }

    fn diagnose_rejected_grammar(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
        init_error: GrammarError,
    ) -> GrammarError {
        if !matches!(
            init_error,
            GrammarError::GrammarMalformed | GrammarError::LazyGrammarMalformed
        ) {
            return init_error;
        }

        let vocab = unsafe { &*model.vocab_ptr() };

        match validate_gbnf(Some(vocab), grammar_str, grammar_root) {
            Ok(()) => init_error,
            Err(GbnfValidationError::RootSymbolMissing { .. }) => GrammarError::RootNotFound,
            Err(rejected) => GrammarError::GrammarRejected(rejected),
        }
    }

    fn sanitize_trigger_patterns(
        trigger_patterns: &[String],
    ) -> Result<Vec<CString>, GrammarError> {
        trigger_patterns
            .iter()
            .map(|pattern| {
                CString::new(pattern.as_str()).map_err(GrammarError::TriggerPatternContainsNul)
            })
            .collect()
    }

    /// # Errors
    /// Returns the [`SamplingError`] describing a sequence breaker llama.cpp would skip, truncate,
    /// or ignore, or the error of initializing the sampler.
    pub fn dry(
        model: &LlamaModel,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        seq_breakers: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) -> Result<Self, SamplingError> {
        let seq_breakers = dry_sequence_breakers(seq_breakers)?;
        let mut seq_breaker_pointers: Vec<*const c_char> = seq_breakers
            .iter()
            .map(|seq_breaker| seq_breaker.as_ptr())
            .collect();

        Self::initialize("DRY", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_dry(
                model.vocab_ptr(),
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                seq_breaker_pointers.as_mut_ptr(),
                seq_breaker_pointers.len(),
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the penalties sampler cannot be initialized.
    pub fn penalties(
        n_vocab: i32,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Result<Self, SamplingError> {
        Self::initialize("penalties", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_penalties(
                n_vocab,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the Mirostat sampler cannot be initialized.
    pub fn mirostat(
        n_vocab: i32,
        seed: u32,
        tau: f32,
        eta: f32,
        m: i32,
    ) -> Result<Self, SamplingError> {
        Self::initialize("Mirostat", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_mirostat(
                n_vocab,
                seed,
                tau,
                eta,
                m,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the Mirostat v2 sampler cannot be initialized.
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Result<Self, SamplingError> {
        Self::initialize("Mirostat v2", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_mirostat_v2(
                seed,
                tau,
                eta,
                out_sampler,
                out_error,
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the distribution sampler cannot be initialized.
    pub fn dist(seed: u32) -> Result<Self, SamplingError> {
        Self::initialize("distribution", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_dist(seed, out_sampler, out_error)
        })
    }

    /// # Errors
    ///
    /// Returns [`SamplingError`] if the greedy sampler cannot be initialized.
    pub fn greedy() -> Result<Self, SamplingError> {
        Self::initialize("greedy", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_greedy(out_sampler, out_error)
        })
    }

    /// # Errors
    /// Returns [`SamplingError::IntegerOverflow`] if `biases.len()` exceeds `i32::MAX`.
    ///
    pub fn logit_bias(n_vocab: i32, biases: &[LlamaLogitBias]) -> Result<Self, SamplingError> {
        let bias_count = checked_usize_as_i32_sampling(biases.len())?;
        let data = biases
            .as_ptr()
            .cast::<llama_cpp_bindings_sys::llama_logit_bias>();

        Self::initialize("logit bias", |out_sampler, out_error| unsafe {
            llama_cpp_bindings_sys::llama_rs_sampler_init_logit_bias(
                n_vocab,
                bias_count,
                data,
                out_sampler,
                out_error,
            )
        })
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_sampler_free(self.sampler.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;

    use llama_cpp_error_recorder::RecordedError;
    use llama_cpp_error_recorder::record;

    use super::LlamaSampler;
    use crate::GrammarError;
    use crate::SampleError;
    use crate::sanitized_grammar::SanitizedGrammar;

    unsafe extern "C" fn record_a_failure_and_exclude_every_candidate(
        _sampler: *mut llama_cpp_bindings_sys::llama_sampler,
        candidates: *mut llama_cpp_bindings_sys::llama_token_data_array,
    ) {
        record(RecordedError::new("mask failed".to_string()));

        let candidates = unsafe { &mut *candidates };
        let candidate_data =
            unsafe { std::slice::from_raw_parts_mut(candidates.data, candidates.size) };
        for candidate in candidate_data {
            candidate.logit = f32::NEG_INFINITY;
        }
    }

    fn greedy_sampler() -> LlamaSampler {
        LlamaSampler::greedy().expect("greedy sampler must initialize")
    }

    fn penalties_sampler() -> LlamaSampler {
        LlamaSampler::penalties(32_000, 64, 1.1, 0.0, 0.0)
            .expect("penalties sampler must initialize")
    }

    static mut FAILING_EXCLUDING_SAMPLER_I: llama_cpp_bindings_sys::llama_sampler_i =
        llama_cpp_bindings_sys::llama_sampler_i {
            name: None,
            accept: None,
            apply: Some(record_a_failure_and_exclude_every_candidate),
            reset: None,
            clone: None,
            free: None,
            backend_init: None,
            backend_accept: None,
            backend_apply: None,
            backend_set_input: None,
            backend_reset: None,
            copy_state: None,
        };

    fn failing_excluding_sampler() -> LlamaSampler {
        let sampler = unsafe {
            llama_cpp_bindings_sys::llama_sampler_init(
                &raw mut FAILING_EXCLUDING_SAMPLER_I,
                ptr::null_mut(),
            )
        };

        LlamaSampler::from_raw(sampler, "failing excluding sampler")
            .expect("the failing excluding sampler must initialize")
    }

    #[test]
    fn a_grammar_callback_failure_takes_precedence_over_a_failed_application() {
        use crate::token::LlamaToken;
        use crate::token::data::LlamaTokenData;
        use crate::token::data_array::LlamaTokenDataArray;

        let mut data_array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 2.0, 0.0),
            ],
            false,
        );

        assert_eq!(
            failing_excluding_sampler().apply(&mut data_array),
            Err(SampleError::GrammarCallbackFailed {
                message: "mask failed".to_string()
            })
        );
    }

    #[test]
    fn null_native_sampler_is_initialization_error() {
        let result = LlamaSampler::from_raw(ptr::null_mut(), "test");

        assert_eq!(
            result.unwrap_err(),
            crate::SamplingError::SamplerUnavailable { sampler: "test" }
        );
    }

    #[test]
    fn sanitize_grammar_strings_valid() {
        assert_eq!(
            LlamaSampler::sanitize_grammar_strings("root ::= \"hello\"", "root"),
            Ok(SanitizedGrammar {
                grammar: CString::new("root ::= \"hello\"").expect("the literal has no nul byte"),
                root: CString::new("root").expect("the literal has no nul byte"),
            })
        );
    }

    #[test]
    fn sanitize_grammar_strings_null_byte_in_root_only() {
        assert_eq!(
            LlamaSampler::sanitize_grammar_strings("root ::= \"hello\"", "ro\0ot"),
            Err(GrammarError::RootContainsNul(
                CString::new("ro\0ot").expect_err("the root carries a nul byte")
            ))
        );
    }

    #[test]
    fn sanitize_grammar_strings_null_byte_in_grammar() {
        assert_eq!(
            LlamaSampler::sanitize_grammar_strings("root ::= \"\0\"", "root"),
            Err(GrammarError::GrammarContainsNul(
                CString::new("root ::= \"\0\"").expect_err("the grammar carries a nul byte")
            ))
        );
    }

    #[test]
    fn sanitize_grammar_strings_null_byte_in_root() {
        assert_eq!(
            LlamaSampler::sanitize_grammar_strings("ro\0ot ::= \"hello\"", "ro\0ot"),
            Err(GrammarError::GrammarContainsNul(
                CString::new("ro\0ot ::= \"hello\"").expect_err("the grammar carries a nul byte")
            )),
            "the grammar is checked before the root, so the grammar reports first"
        );
    }

    #[test]
    fn sanitize_trigger_patterns_valid() {
        let patterns = vec!["^hello$".to_string(), "world.*".to_string()];
        let result = LlamaSampler::sanitize_trigger_patterns(&patterns);

        assert!(result.is_ok());
        assert_eq!(result.expect("valid trigger patterns").len(), 2);
    }

    #[test]
    fn sanitize_trigger_patterns_empty_list() {
        let patterns: Vec<String> = vec![];
        let result = LlamaSampler::sanitize_trigger_patterns(&patterns);

        assert!(result.is_ok());
        assert_eq!(
            result.expect("valid trigger patterns"),
            Vec::<CString>::new()
        );
    }

    #[test]
    fn sanitize_trigger_patterns_null_byte() {
        let patterns = vec!["hel\0lo".to_string()];
        assert_eq!(
            LlamaSampler::sanitize_trigger_patterns(&patterns),
            Err(GrammarError::TriggerPatternContainsNul(
                CString::new("hel\0lo").expect_err("the pattern carries a nul byte")
            ))
        );
    }

    #[test]
    fn apply_modifies_data_array() {
        use crate::token::LlamaToken;
        use crate::token::data::LlamaTokenData;
        use crate::token::data_array::LlamaTokenDataArray;

        let sampler = greedy_sampler();
        let mut data_array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            ],
            false,
        );

        assert!(sampler.apply(&mut data_array).is_ok());

        assert_eq!(data_array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn apply_surfaces_a_sampler_that_excludes_every_candidate() {
        use crate::error::sampler_apply_error::SamplerApplyError;
        use crate::token::LlamaToken;
        use crate::token::data::LlamaTokenData;
        use crate::token::data_array::LlamaTokenDataArray;
        use crate::token::logit_bias::LlamaLogitBias;

        let excluding_sampler = LlamaSampler::logit_bias(
            2,
            &[
                LlamaLogitBias::new(LlamaToken::new(0), f32::NEG_INFINITY),
                LlamaLogitBias::new(LlamaToken::new(1), f32::NEG_INFINITY),
            ],
        )
        .expect("the logit bias sampler must initialize");
        let mut data_array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 2.0, 0.0),
            ],
            false,
        );

        assert_eq!(
            excluding_sampler.apply(&mut data_array),
            Err(SampleError::SamplerApply(
                SamplerApplyError::EveryCandidateExcluded
            ))
        );
    }

    #[test]
    fn accept_succeeds() {
        let mut sampler = LlamaSampler::chain_simple([penalties_sampler(), greedy_sampler()])
            .expect("sampler chain must initialize");

        sampler
            .accept(crate::token::LlamaToken::new(1))
            .expect("test: accept should succeed");
    }

    #[test]
    fn accept_succeeds_on_penalties_sampler() {
        let mut sampler = LlamaSampler::chain_simple([penalties_sampler(), greedy_sampler()])
            .expect("sampler chain must initialize");

        let result = sampler.accept(crate::token::LlamaToken::new(42));

        assert!(result.is_ok());
    }

    #[test]
    fn accept_many_multiple_tokens() {
        use crate::token::LlamaToken;

        let mut sampler = LlamaSampler::chain_simple([penalties_sampler(), greedy_sampler()])
            .expect("sampler chain must initialize");

        sampler
            .accept_many([LlamaToken::new(1), LlamaToken::new(2), LlamaToken::new(3)])
            .expect("test: accept_many should succeed");
    }

    #[test]
    fn with_tokens_builder_pattern() {
        use crate::token::LlamaToken;

        let _sampler = LlamaSampler::chain_simple([penalties_sampler(), greedy_sampler()])
            .expect("sampler chain must initialize")
            .with_tokens([LlamaToken::new(10), LlamaToken::new(20)])
            .expect("test: with_tokens should succeed");
    }

    #[test]
    fn all_sampler_constructors() {
        use crate::token::LlamaToken;
        use crate::token::logit_bias::LlamaLogitBias;

        let _temp = LlamaSampler::temp(0.8).expect("temperature sampler must initialize");
        let _temp_ext = LlamaSampler::temp_ext(0.8, 0.1, 1.0)
            .expect("extended temperature sampler must initialize");
        let _top_k = LlamaSampler::top_k(40).expect("top-k sampler must initialize");
        let _top_n_sigma =
            LlamaSampler::top_n_sigma(2.0).expect("top-n-sigma sampler must initialize");
        let _top_p = LlamaSampler::top_p(0.9, 1).expect("top-p sampler must initialize");
        let _min_p = LlamaSampler::min_p(0.05, 1).expect("min-p sampler must initialize");
        let _typical = LlamaSampler::typical(0.9, 1).expect("typical sampler must initialize");
        let _xtc = LlamaSampler::xtc(0.1, 0.5, 1, 42).expect("XTC sampler must initialize");
        let _dist = LlamaSampler::dist(42).expect("distribution sampler must initialize");
        let _mirostat = LlamaSampler::mirostat(32000, 42, 5.0, 0.1, 100)
            .expect("Mirostat sampler must initialize");
        let _mirostat_v2 =
            LlamaSampler::mirostat_v2(42, 5.0, 0.1).expect("Mirostat v2 sampler must initialize");
        let biases = vec![LlamaLogitBias::new(LlamaToken::new(0), -100.0)];
        let _logit_bias = LlamaSampler::logit_bias(32000, &biases);
        let _chain =
            LlamaSampler::chain([greedy_sampler()], true).expect("sampler chain must initialize");
    }

    #[test]
    fn reset_and_get_seed() {
        let mut sampler = LlamaSampler::dist(42).expect("distribution sampler must initialize");
        assert!(sampler.reset().is_ok());
        let _seed = sampler.get_seed();
    }

    #[test]
    fn debug_formatting() {
        let sampler = greedy_sampler();
        let debug_output = format!("{sampler:?}");
        assert_eq!(debug_output, "LlamaSampler { .. }");
    }
}

#[cfg(test)]
mod checked_usize_as_i32_sampling_tests {
    use super::checked_usize_as_i32_sampling;

    #[test]
    fn checked_usize_as_i32_sampling_overflow() {
        let result = checked_usize_as_i32_sampling(usize::MAX);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod grammar_callback_error_to_result_tests {
    use llama_cpp_error_recorder::RecordedError;

    use super::grammar_callback_error_to_result;
    use crate::SampleError;

    #[test]
    fn grammar_callback_error_to_result_maps_recorded_error() {
        let result =
            grammar_callback_error_to_result(Some(RecordedError::new("mask failed".to_string())));

        assert_eq!(
            result.unwrap_err(),
            SampleError::GrammarCallbackFailed {
                message: "mask failed".to_string()
            }
        );
    }

    #[test]
    fn grammar_callback_error_to_result_maps_absence_to_ok() {
        assert!(grammar_callback_error_to_result(None).is_ok());
    }
}

#[cfg(test)]
mod grammar_callback_error_to_accept_result_tests {
    use llama_cpp_error_recorder::RecordedError;

    use super::grammar_callback_error_to_accept_result;
    use crate::SamplerAcceptError;

    #[test]
    fn grammar_callback_error_to_accept_result_maps_recorded_error() {
        let result = grammar_callback_error_to_accept_result(Some(RecordedError::new(
            "consume failed".to_string(),
        )));

        assert_eq!(
            result,
            Err(SamplerAcceptError::GrammarCallbackFailed {
                message: "consume failed".to_string()
            })
        );
    }

    #[test]
    fn grammar_callback_error_to_accept_result_maps_absence_to_ok() {
        assert!(grammar_callback_error_to_accept_result(None).is_ok());
    }
}

#[cfg(test)]
mod check_sampler_accept_status_tests {
    use std::ptr;

    use super::check_sampler_accept_status;
    use crate::FfiContractError;
    use crate::FfiStatusError;
    use crate::SamplerAcceptError;

    #[test]
    fn check_sampler_accept_status_ok() {
        let result = check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_OK,
            ptr::null_mut(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn check_sampler_accept_status_exception_maps_to_typed_variant() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"grammar state corrupted".as_ptr())
        };

        assert_eq!(
            check_sampler_accept_status(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                out_error,
            ),
            Err(SamplerAcceptError::GrammarStateCorrupted {
                message: "grammar state corrupted".to_owned(),
            })
        );
    }

    #[test]
    fn check_sampler_accept_status_allocation_failure_maps_to_not_enough_memory() {
        let result = check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(SamplerAcceptError::NotEnoughMemory));
    }

    #[test]
    fn sampler_accept_null_sampler_status_is_a_contract_error() {
        let status = llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG;
        let result = check_sampler_accept_status(status, ptr::null_mut());

        assert_eq!(
            result,
            Err(SamplerAcceptError::FfiContract(FfiContractError {
                operation: "llama_rs_sampler_accept",
                detail: "was given a null sampler argument",
            }))
        );
    }

    #[test]
    fn check_sampler_accept_status_maps_null_out_error_arg() {
        let outcome = check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_NULL_OUT_ERROR_ARG,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_sampler_accept",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn check_sampler_accept_status_maps_llama_cpp_out_of_memory() {
        let outcome = check_sampler_accept_status(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(SamplerAcceptError::LlamaCppOutOfMemory));
    }

    #[test]
    fn check_sampler_accept_status_exception_without_a_message_is_a_contract_error() {
        assert_eq!(
            check_sampler_accept_status(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(SamplerAcceptError::FfiContract(FfiContractError {
                operation: "llama_rs_sampler_accept",
                detail: "reported a thrown C++ exception without an error message",
            }))
        );
    }

    #[test]
    fn check_sampler_accept_status_maps_an_unknown_status() {
        assert_eq!(
            check_sampler_accept_status(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_THREW_CXX_EXCEPTION + 1,
                ptr::null_mut(),
            ),
            Err(SamplerAcceptError::FfiStatus(FfiStatusError {
                operation: "llama_rs_sampler_accept",
                code: i64::from(
                    llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_ACCEPT_LLAMA_CPP_THREW_CXX_EXCEPTION
                        + 1
                ),
            }))
        );
    }
}

#[cfg(test)]
mod sampler_sample_status_to_result_tests {
    use std::ptr;

    use super::sampler_sample_status_to_result;
    use crate::FfiStatusError;
    use crate::SampleError;

    #[test]
    fn sampler_sample_status_allocation_failure_maps_to_not_enough_memory() {
        let result = sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED,
            -1,
            ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), SampleError::NotEnoughMemory);
    }

    #[test]
    fn sampler_sample_status_exception_without_a_message_is_a_contract_error() {
        let result = sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            -1,
            ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            crate::FfiContractError {
                operation: "llama_rs_sampler_sample",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn sampler_sample_null_context_status_is_a_contract_error() {
        let status = llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG;
        let result = sampler_sample_status_to_result(status, -1, ptr::null_mut());

        assert_eq!(
            result,
            Err(SampleError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_sampler_sample",
                detail: "was given a null ctx argument",
            }))
        );
    }

    #[test]
    fn sampler_sample_status_to_result_maps_null_sampler_arg() {
        let outcome = sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_SAMPLER_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_sample",
                    detail: "was given a null sampler argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_sample_status_to_result_maps_null_out_token_arg() {
        let outcome = sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_TOKEN_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_sample",
                    detail: "was given a null out_token argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_sample_status_to_result_maps_null_out_error_arg() {
        let outcome = sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_ERROR_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_sample",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_sample_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = sampler_sample_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(SampleError::LlamaCppOutOfMemory));
    }

    #[test]
    fn sampler_sample_status_exception_carries_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"sampler threw".as_ptr()) };

        assert_eq!(
            sampler_sample_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                -1,
                out_error,
            ),
            Err(SampleError::Reported {
                message: "sampler threw".to_owned(),
            })
        );
    }

    #[test]
    fn sampler_sample_status_to_result_maps_an_unknown_status() {
        assert_eq!(
            sampler_sample_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_THREW_CXX_EXCEPTION + 1,
                -1,
                ptr::null_mut(),
            ),
            Err(SampleError::FfiStatus(FfiStatusError {
                operation: "llama_rs_sampler_sample",
                code: i64::from(
                    llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_SAMPLE_LLAMA_CPP_THREW_CXX_EXCEPTION
                        + 1
                ),
            }))
        );
    }
}

#[cfg(test)]
mod sampler_init_status_to_result_tests {
    use std::ptr;

    use super::sampler_init_status_to_result;

    #[test]
    fn sampler_init_status_null_maps_to_sampler_unavailable() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_RETURNED_NULL,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(crate::SamplingError::SamplerUnavailable {
                sampler: "penalties"
            })
        );
    }

    #[test]
    fn sampler_init_status_allocation_failure_maps_to_not_enough_memory() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(crate::SamplingError::NotEnoughMemory)
        );
    }

    #[test]
    fn sampler_init_status_out_of_memory_names_the_sampler() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(crate::SamplingError::LlamaCppOutOfMemory {
                sampler: "penalties"
            })
        );
    }

    #[test]
    fn sampler_init_status_exception_carries_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"vector too long".as_ptr()) };

        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
                "penalties",
            )
            .err(),
            Some(crate::SamplingError::Reported {
                sampler: "penalties",
                message: "vector too long".to_owned(),
            })
        );
    }

    #[test]
    fn sampler_init_status_to_result_maps_null_out_sampler_arg() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_NULL_OUT_SAMPLER_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_init",
                    detail: "was given a null out_sampler argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_init_status_to_result_maps_null_out_error_arg() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_NULL_OUT_ERROR_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_init",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_init_status_to_result_maps_llama_cpp_threw_cxx_exception_without_a_message() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_init",
                    detail: "reported a thrown C++ exception without an error message",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_init_status_to_result_maps_an_unknown_status() {
        assert_eq!(
            sampler_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION + 1,
                ptr::null_mut(),
                ptr::null_mut(),
                "penalties",
            )
            .err(),
            Some(
                crate::FfiStatusError {
                    operation: "llama_rs_sampler_init",
                    code: i64::from(
                        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION
                            + 1
                    ),
                }
                .into()
            )
        );
    }
}

#[cfg(test)]
mod sampler_chain_add_status_to_result_tests {
    use std::ptr;

    use super::sampler_chain_add_status_to_result;

    #[test]
    fn sampler_chain_add_status_allocation_failure_maps_to_not_enough_memory() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(crate::SamplingError::NotEnoughMemory)
        );
    }

    #[test]
    fn sampler_chain_add_status_out_of_memory_names_the_chain() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
            ),
            Err(crate::SamplingError::LlamaCppOutOfMemory { sampler: "chain" })
        );
    }

    #[test]
    fn sampler_chain_add_status_exception_carries_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"vector too long".as_ptr()) };

        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION,
                out_error,
            ),
            Err(crate::SamplingError::Reported {
                sampler: "chain",
                message: "vector too long".to_owned(),
            })
        );
    }

    #[test]
    fn sampler_chain_add_status_to_result_maps_null_chain_arg() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_CHAIN_ARG,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "was given a null chain argument",
            }
            .into())
        );
    }

    #[test]
    fn sampler_chain_add_status_to_result_maps_null_sampler_arg() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_SAMPLER_ARG,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "was given a null sampler argument",
            }
            .into())
        );
    }

    #[test]
    fn sampler_chain_add_status_to_result_maps_null_out_error_arg() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_OUT_ERROR_ARG,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "was given a null out_error argument",
            }
            .into())
        );
    }

    #[test]
    fn sampler_chain_add_status_to_result_maps_llama_cpp_threw_cxx_exception_without_a_message() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_chain_add",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn sampler_chain_add_status_to_result_maps_an_unknown_status() {
        assert_eq!(
            sampler_chain_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION
                    + 1,
                ptr::null_mut(),
            ),
            Err(crate::FfiStatusError {
                operation: "llama_rs_sampler_chain_add",
                code: i64::from(
                    llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION
                        + 1
                ),
            }
            .into())
        );
    }
}

#[cfg(test)]
mod sampler_init_grammar_status_to_result_tests {
    use std::ptr;

    use super::sampler_init_grammar_status_to_result;
    use crate::FfiStatusError;
    use crate::GrammarError;

    #[test]
    fn sampler_init_grammar_status_null_maps_to_grammar_malformed() {
        let result = sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_RETURNED_NULL,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::GrammarMalformed);
    }

    #[test]
    fn sampler_init_grammar_status_allocation_failure_maps_to_not_enough_memory() {
        let result = sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::NotEnoughMemory);
    }

    #[test]
    fn sampler_init_grammar_status_exception_without_a_message_is_a_contract_error() {
        let result = sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            crate::FfiContractError {
                operation: "llama_rs_sampler_init_grammar",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn grammar_null_output_argument_status_is_a_contract_error() {
        let status = llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_SAMPLER_ARG;
        let result =
            sampler_init_grammar_status_to_result(status, ptr::null_mut(), ptr::null_mut());

        assert_eq!(
            result.unwrap_err(),
            GrammarError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_sampler_init_grammar",
                detail: "was given a null out_sampler argument",
            })
        );
    }

    #[test]
    fn sampler_init_grammar_status_to_result_maps_null_out_error_arg() {
        let outcome = sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_ERROR_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_init_grammar",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_init_grammar_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = sampler_init_grammar_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(GrammarError::LlamaCppOutOfMemory));
    }

    #[test]
    fn sampler_init_grammar_status_exception_carries_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"grammar threw".as_ptr()) };

        assert_eq!(
            sampler_init_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
            .err(),
            Some(GrammarError::Reported {
                message: "grammar threw".to_owned(),
            })
        );
    }

    #[test]
    fn sampler_init_grammar_status_to_result_maps_an_unknown_status() {
        assert_eq!(
            sampler_init_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION
                    + 1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
            .err(),
            Some(GrammarError::FfiStatus(FfiStatusError {
                operation: "llama_rs_sampler_init_grammar",
                code: i64::from(
                    llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION
                        + 1
                ),
            }))
        );
    }
}

#[cfg(test)]
mod sampler_init_grammar_lazy_patterns_status_to_result_tests {
    use std::ptr;

    use super::sampler_init_grammar_lazy_patterns_status_to_result;
    use crate::FfiStatusError;
    use crate::GrammarError;

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_null_maps_to_lazy_patterns_grammar_malformed() {
        let result = sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_RETURNED_NULL,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::LazyGrammarMalformed);
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_allocation_failure_maps_to_not_enough_memory() {
        let result = sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(result.unwrap_err(), GrammarError::NotEnoughMemory);
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_exception_without_a_message_is_a_contract_error() {
        let result = sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_THREW_CXX_EXCEPTION,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            crate::FfiContractError {
                operation: "llama_rs_sampler_init_grammar_lazy_patterns",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn lazy_grammar_null_output_argument_status_is_a_contract_error() {
        let status = llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_SAMPLER_ARG;
        let result = sampler_init_grammar_lazy_patterns_status_to_result(
            status,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        assert_eq!(
            result.unwrap_err(),
            GrammarError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_sampler_init_grammar_lazy_patterns",
                detail: "was given a null out_sampler argument",
            })
        );
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_to_result_maps_null_out_error_arg() {
        let outcome = sampler_init_grammar_lazy_patterns_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_ERROR_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_sampler_init_grammar_lazy_patterns",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = sampler_init_grammar_lazy_patterns_status_to_result(llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_OUT_OF_MEMORY, ptr::null_mut(), ptr::null_mut());
        assert_eq!(outcome.err(), Some(GrammarError::LlamaCppOutOfMemory));
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_exception_carries_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"lazy grammar threw".as_ptr()) };

        assert_eq!(
            sampler_init_grammar_lazy_patterns_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
            .err(),
            Some(GrammarError::Reported {
                message: "lazy grammar threw".to_owned(),
            })
        );
    }

    #[test]
    fn sampler_init_grammar_lazy_patterns_status_to_result_maps_an_unknown_status() {
        let unknown_status =
            llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_LLAMA_CPP_THREW_CXX_EXCEPTION
                + 1;

        assert_eq!(
            sampler_init_grammar_lazy_patterns_status_to_result(
                unknown_status,
                ptr::null_mut(),
                ptr::null_mut(),
            )
            .err(),
            Some(GrammarError::FfiStatus(FfiStatusError {
                operation: "llama_rs_sampler_init_grammar_lazy_patterns",
                code: i64::from(unknown_status),
            }))
        );
    }
}
