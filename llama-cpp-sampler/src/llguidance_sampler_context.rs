use llguidance::api::TopLevelGrammar;
use llguidance::earley::SlicedBiasComputer;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr;
use std::slice;
use std::sync::Arc;
use toktrie::ApproximateTokEnv;
use toktrie::InferenceCapabilities;

use llama_cpp_error_recorder::record::record;

use crate::grammar_matcher::GrammarMatcher;
use crate::mask_outcome::MaskOutcome;
use crate::sampler_error::SamplerError;

const unsafe extern "C" fn llg_name(
    _smpl: *const llama_cpp_bindings_sys::bindings::llama_sampler,
) -> *const c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut llama_cpp_bindings_sys::bindings::llama_sampler,
    token: llama_cpp_bindings_sys::bindings::llama_token,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlguidanceSamplerContext>() };

    if let Err(grammar_error) = ctx.grammar.consume_token(token.cast_unsigned()) {
        record(grammar_error);
    }
}

unsafe extern "C" fn llg_apply(
    smpl: *mut llama_cpp_bindings_sys::bindings::llama_sampler,
    cur_p: *mut llama_cpp_bindings_sys::bindings::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlguidanceSamplerContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let mask = match ctx.grammar.compute_mask() {
        Ok(MaskOutcome::Constrained(mask)) => mask,
        Ok(MaskOutcome::GrammarComplete) => return,
        Err(grammar_error) => {
            record(grammar_error);

            return;
        }
    };

    let data = unsafe { slice::from_raw_parts_mut(cur_p.data, cur_p.size) };
    for item in data.iter_mut() {
        if !mask.is_allowed(item.id.cast_unsigned()) {
            item.logit = f32::NEG_INFINITY;
        }
    }
}

unsafe extern "C" fn llg_reset(smpl: *mut llama_cpp_bindings_sys::bindings::llama_sampler) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlguidanceSamplerContext>() };

    if let Err(grammar_error) = ctx.grammar.reset() {
        record(grammar_error);
    }
}

unsafe extern "C" fn llg_clone(
    smpl: *const llama_cpp_bindings_sys::bindings::llama_sampler,
) -> *mut llama_cpp_bindings_sys::bindings::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlguidanceSamplerContext>() };
    let new_ctx = Box::new(LlguidanceSamplerContext {
        grammar: ctx.grammar.deep_clone(),
        tok_env: Arc::clone(&ctx.tok_env),
        grammar_kind: ctx.grammar_kind.clone(),
        grammar_data: ctx.grammar_data.clone(),
    });
    unsafe {
        llama_cpp_bindings_sys::bindings::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(new_ctx).cast::<c_void>(),
        )
    }
}

unsafe extern "C" fn llg_free(smpl: *mut llama_cpp_bindings_sys::bindings::llama_sampler) {
    drop(unsafe { Box::from_raw((*smpl).ctx.cast::<LlguidanceSamplerContext>()) });
}

static mut LLG_SAMPLER_I: llama_cpp_bindings_sys::bindings::llama_sampler_i =
    llama_cpp_bindings_sys::bindings::llama_sampler_i {
        name: Some(llg_name),
        accept: Some(llg_accept),
        apply: Some(llg_apply),
        reset: Some(llg_reset),
        clone: Some(llg_clone),
        free: Some(llg_free),
        backend_init: None,
        backend_accept: None,
        backend_apply: None,
        backend_set_input: None,
        backend_reset: None,
        copy_state: None,
    };

const UNCOMPILABLE_BIAS_SLICE: &str = "(";

fn bias_slices() -> Vec<String> {
    let factory_fault_reached = unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_fault_point_reached(
            llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_LLGUIDANCE_FACTORY_FAILS,
        )
    };

    if factory_fault_reached {
        return vec![UNCOMPILABLE_BIAS_SLICE.to_owned()];
    }

    SlicedBiasComputer::general_slices()
}

pub struct LlguidanceSamplerContext {
    grammar: GrammarMatcher,
    tok_env: Arc<ApproximateTokEnv>,
    grammar_kind: String,
    grammar_data: String,
}

impl LlguidanceSamplerContext {
    /// # Errors
    ///
    /// Returns [`SamplerError`] if the parser factory, grammar, or parser cannot be created.
    pub fn new(
        tok_env: Arc<ApproximateTokEnv>,
        grammar_kind: &str,
        grammar_data: &str,
    ) -> Result<Self, SamplerError> {
        let tok_env_dyn: Arc<dyn toktrie::TokenizerEnv + Sync> = tok_env.clone();

        let factory = llguidance::ParserFactory::new(
            &tok_env_dyn,
            InferenceCapabilities::default(),
            &bias_slices(),
        )
        .map_err(|factory_error| SamplerError::LlguidanceFactoryUnavailable {
            message: factory_error.to_string(),
        })?;

        let grammar = TopLevelGrammar::from_tagged_str(grammar_kind, grammar_data).map_err(
            |parse_error| SamplerError::LlguidanceGrammarInvalid {
                message: parse_error.to_string(),
            },
        )?;

        let parser = factory.create_parser(grammar).map_err(|parser_error| {
            SamplerError::LlguidanceParserUnavailable {
                message: parser_error.to_string(),
            }
        })?;

        Ok(Self {
            grammar: GrammarMatcher::new(parser),
            tok_env,
            grammar_kind: grammar_kind.to_string(),
            grammar_data: grammar_data.to_string(),
        })
    }

    /// Hands the context over to `sampler`, which frees it when llama.cpp frees the sampler.
    ///
    /// # Safety
    ///
    /// `sampler` must be a sampler [`Self::initialize_sampler`] initialized that holds no
    /// context yet.
    pub unsafe fn attach_to(self, sampler: *mut llama_cpp_bindings_sys::bindings::llama_sampler) {
        unsafe { (*sampler).ctx = Box::into_raw(Box::new(self)).cast::<c_void>() };
    }

    /// Initializes a llama.cpp sampler that runs the llguidance callbacks and holds no context
    /// yet.
    ///
    /// # Safety
    ///
    /// `out_sampler` and `out_error` must be valid for writes.
    pub unsafe fn initialize_sampler(
        out_sampler: *mut *mut llama_cpp_bindings_sys::bindings::llama_sampler,
        out_error: *mut *mut c_char,
    ) -> llama_cpp_bindings_sys::bindings::llama_rs_sampler_init_status {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_sampler_init(
                &raw mut LLG_SAMPLER_I,
                ptr::null_mut(),
                out_sampler,
                out_error,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use toktrie::ApproximateTokEnv;
    use toktrie::INVALID_TOKEN;
    use toktrie::InferenceCapabilities;
    use toktrie::TokRxInfo;
    use toktrie::TokTrie;

    use llama_cpp_fault_injection::armed_fault::ArmedFault;

    use super::LlguidanceSamplerContext;
    use super::UNCOMPILABLE_BIAS_SLICE;
    use crate::sampler_error::SamplerError;

    #[test]
    fn creating_a_sampler_reports_a_parser_factory_llguidance_cannot_build() {
        let vocabulary_without_end_of_generation = TokRxInfo::new(2, INVALID_TOKEN);
        let letters = Arc::new(ApproximateTokEnv::new(TokTrie::from(
            &vocabulary_without_end_of_generation,
            &[b"a".to_vec(), b"b".to_vec()],
        )));
        let letters_env: Arc<dyn toktrie::TokenizerEnv + Sync> = letters.clone();
        let factory_failure = llguidance::ParserFactory::new(
            &letters_env,
            InferenceCapabilities::default(),
            &[UNCOMPILABLE_BIAS_SLICE.to_owned()],
        )
        .err()
        .map(|factory_error| SamplerError::LlguidanceFactoryUnavailable {
            message: factory_error.to_string(),
        });
        let _fault = ArmedFault::at(
            llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_LLGUIDANCE_FACTORY_FAILS,
        );

        assert_ne!(factory_failure, None);
        assert_eq!(
            LlguidanceSamplerContext::new(letters, "regex", "a").err(),
            factory_failure
        );
    }
}
