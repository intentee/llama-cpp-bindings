use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr;
use std::slice;
use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::earley::SlicedBiasComputer;
use toktrie::ApproximateTokEnv;
use toktrie::InferenceCapabilities;

use llama_cpp_error_recorder::RecordedError;
use llama_cpp_error_recorder::record;

use crate::error::grammar_error::GrammarError;
use crate::grammar_matcher::GrammarMatcher;
use crate::mask_outcome::MaskOutcome;
use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;

struct LlgContext {
    grammar: GrammarMatcher,
    tok_env: Arc<ApproximateTokEnv>,
    grammar_kind: String,
    grammar_data: String,
}

const unsafe extern "C" fn llg_name(
    _smpl: *const llama_cpp_bindings_sys::llama_sampler,
) -> *const c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut llama_cpp_bindings_sys::llama_sampler,
    token: llama_cpp_bindings_sys::llama_token,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };

    if let Err(grammar_error) = ctx.grammar.consume_token(token.cast_unsigned()) {
        record(RecordedError::new(grammar_error));
    }
}

unsafe extern "C" fn llg_apply(
    smpl: *mut llama_cpp_bindings_sys::llama_sampler,
    cur_p: *mut llama_cpp_bindings_sys::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let mask = match ctx.grammar.compute_mask() {
        Ok(MaskOutcome::Constrained(mask)) => mask,
        Ok(MaskOutcome::GrammarComplete) => return,
        Err(grammar_error) => {
            record(RecordedError::new(grammar_error));

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

unsafe extern "C" fn llg_reset(smpl: *mut llama_cpp_bindings_sys::llama_sampler) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };

    if let Err(grammar_error) = ctx.grammar.reset() {
        record(RecordedError::new(grammar_error));
    }
}

unsafe extern "C" fn llg_clone(
    smpl: *const llama_cpp_bindings_sys::llama_sampler,
) -> *mut llama_cpp_bindings_sys::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlgContext>() };
    let new_ctx = Box::new(LlgContext {
        grammar: ctx.grammar.deep_clone(),
        tok_env: Arc::clone(&ctx.tok_env),
        grammar_kind: ctx.grammar_kind.clone(),
        grammar_data: ctx.grammar_data.clone(),
    });
    unsafe {
        llama_cpp_bindings_sys::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(new_ctx).cast::<c_void>(),
        )
    }
}

unsafe extern "C" fn llg_free(smpl: *mut llama_cpp_bindings_sys::llama_sampler) {
    drop(unsafe { Box::from_raw((*smpl).ctx.cast::<LlgContext>()) });
}

static mut LLG_SAMPLER_I: llama_cpp_bindings_sys::llama_sampler_i =
    llama_cpp_bindings_sys::llama_sampler_i {
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
        llama_cpp_bindings_sys::llama_rs_fault_point_reached(
            llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_LLGUIDANCE_FACTORY_FAILS,
        )
    };

    if factory_fault_reached {
        return vec![UNCOMPILABLE_BIAS_SLICE.to_owned()];
    }

    SlicedBiasComputer::general_slices()
}

fn create_llg_sampler_from_tok_env(
    tok_env: Arc<ApproximateTokEnv>,
    grammar_kind: &str,
    grammar_data: &str,
) -> Result<LlamaSampler, GrammarError> {
    let tok_env_dyn: Arc<dyn toktrie::TokenizerEnv + Sync> = tok_env.clone();

    let factory = llguidance::ParserFactory::new(
        &tok_env_dyn,
        InferenceCapabilities::default(),
        &bias_slices(),
    )
    .map_err(|factory_error| GrammarError::LlguidanceFactoryUnavailable {
        message: factory_error.to_string(),
    })?;

    let grammar =
        TopLevelGrammar::from_tagged_str(grammar_kind, grammar_data).map_err(|parse_error| {
            GrammarError::LlguidanceGrammarInvalid {
                message: parse_error.to_string(),
            }
        })?;

    let parser = factory.create_parser(grammar).map_err(|parser_error| {
        GrammarError::LlguidanceParserUnavailable {
            message: parser_error.to_string(),
        }
    })?;

    let ctx = Box::new(LlgContext {
        grammar: GrammarMatcher::new(parser),
        tok_env,
        grammar_kind: grammar_kind.to_string(),
        grammar_data: grammar_data.to_string(),
    });

    LlamaSampler::initialize("llguidance", |out_sampler, out_error| unsafe {
        llama_cpp_bindings_sys::llama_rs_sampler_init(
            &raw mut LLG_SAMPLER_I,
            ptr::null_mut(),
            out_sampler,
            out_error,
        )
    })
    .inspect(|sampler| {
        unsafe { (*sampler.as_ptr()).ctx = Box::into_raw(ctx).cast::<c_void>() };
    })
    .map_err(Into::into)
}

/// # Errors
///
/// Returns `GrammarError` if the parser factory, grammar, or parser cannot be created.
pub fn create_llg_sampler(
    model: &LlamaModel,
    grammar_kind: &str,
    grammar_data: &str,
) -> Result<LlamaSampler, GrammarError> {
    create_llg_sampler_from_tok_env(model.approximate_tok_env()?, grammar_kind, grammar_data)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use toktrie::ApproximateTokEnv;
    use toktrie::INVALID_TOKEN;
    use toktrie::InferenceCapabilities;
    use toktrie::TokRxInfo;
    use toktrie::TokTrie;

    use llama_cpp_fault_injection::ArmedFault;

    use super::UNCOMPILABLE_BIAS_SLICE;
    use super::create_llg_sampler_from_tok_env;
    use crate::error::grammar_error::GrammarError;
    use crate::error::grammar_runtime_error::GrammarRuntimeError;
    use crate::error::sample_error::SampleError;
    use crate::error::sampler_accept_error::SamplerAcceptError;
    use crate::sampling::LlamaSampler;
    use crate::token::LlamaToken;
    use crate::token::data::LlamaTokenData;
    use crate::token::data_array::LlamaTokenDataArray;

    const LETTER_A: LlamaToken = LlamaToken::new(0);
    const LETTER_B: LlamaToken = LlamaToken::new(1);
    const OUT_OF_VOCABULARY: LlamaToken = LlamaToken::new(2);

    fn letters() -> Arc<ApproximateTokEnv> {
        let vocabulary_without_end_of_generation = TokRxInfo::new(2, INVALID_TOKEN);

        Arc::new(ApproximateTokEnv::new(TokTrie::from(
            &vocabulary_without_end_of_generation,
            &[b"a".to_vec(), b"b".to_vec()],
        )))
    }

    fn letter_a_sampler() -> Result<LlamaSampler, GrammarError> {
        create_llg_sampler_from_tok_env(letters(), "regex", "a")
    }

    fn letter_candidates() -> LlamaTokenDataArray {
        LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LETTER_A, 1.0, 0.0),
                LlamaTokenData::new(LETTER_B, 2.0, 0.0),
            ],
            false,
        )
    }

    fn out_of_vocabulary_failure() -> String {
        GrammarRuntimeError::InternalParserError {
            message: "token id 2 out of range".to_owned(),
        }
        .to_string()
    }

    #[test]
    fn apply_leaves_candidates_untouched_once_the_grammar_is_complete() {
        let mut sampler = letter_a_sampler().unwrap();
        let mut candidates = letter_candidates();

        assert_eq!(sampler.accept(LETTER_A), Ok(()));
        assert_eq!(sampler.apply(&mut candidates), Ok(()));
        assert_eq!(candidates, letter_candidates());
    }

    #[test]
    fn apply_reports_a_grammar_in_an_error_state_as_a_callback_failure() {
        let mut sampler = letter_a_sampler().unwrap();

        assert_eq!(
            sampler.accept(OUT_OF_VOCABULARY),
            Err(SamplerAcceptError::GrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
        );
        assert_eq!(
            sampler.apply(&mut letter_candidates()),
            Err(SampleError::GrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
        );
    }

    #[test]
    fn accepting_many_tokens_stops_at_the_first_one_the_grammar_rejects() {
        let mut sampler = letter_a_sampler().unwrap();

        assert_eq!(
            sampler.accept_many([OUT_OF_VOCABULARY, LETTER_A]),
            Err(SamplerAcceptError::GrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
        );
    }

    #[test]
    fn reset_reports_a_grammar_in_an_error_state_as_a_callback_failure() {
        let mut sampler = letter_a_sampler().unwrap();

        assert_eq!(sampler.accept(LETTER_A), Ok(()));
        assert_eq!(
            sampler.accept(OUT_OF_VOCABULARY),
            Err(SamplerAcceptError::GrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
        );
        assert_eq!(
            sampler.reset(),
            Err(SampleError::GrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
        );
    }

    #[test]
    fn creating_a_sampler_reports_a_parser_factory_llguidance_cannot_build() {
        let letters_env: Arc<dyn toktrie::TokenizerEnv + Sync> = letters();
        let factory_failure = llguidance::ParserFactory::new(
            &letters_env,
            InferenceCapabilities::default(),
            &[UNCOMPILABLE_BIAS_SLICE.to_owned()],
        )
        .err()
        .map(|factory_error| GrammarError::LlguidanceFactoryUnavailable {
            message: factory_error.to_string(),
        });
        let _fault =
            ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_LLGUIDANCE_FACTORY_FAILS);

        assert_ne!(factory_failure, None);
        assert_eq!(letter_a_sampler().err(), factory_failure);
    }
}
