use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_sys::{llama_pos, llama_rs_batch_entry, llama_seq_id};
use llama_cpp_ffi_status::wrapper_operation::WrapperOperation;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::sampled_token::SampledToken;

use crate::context_error::ContextError;
use crate::context_limits::ContextLimits;
use crate::decision_order::DecisionOrder;

fn batch_ext_assign_entries_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_batch_ext_assign_entries_status,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), ContextError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_batch_ext_assign_entries");

    let refused_validated_entry = |detail: &'static str| -> Result<(), ContextError> {
        Err(OPERATION.contract_violated(detail).into())
    };

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextError::BatchTransferNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextError::BatchTransferLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                OPERATION.read_and_free_cpp_string(out_error, "reported a thrown C++ exception without an error message")
            }?;
            Err(ContextError::BatchTransferReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_BATCH_FULL => refused_validated_entry(
            "refused an entry of a batch validated against its limits as full",
        ),
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_INVALID_TOKEN => {
            refused_validated_entry("refused a token validated against the vocabulary")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_INVALID_SEQUENCE_ID => {
            refused_validated_entry("refused a sequence id validated against the context")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_RETURNED_UNKNOWN_CODE => {
            Err(WrapperOperation::new("llama_batch_ext_add_token").unknown_status(i64::from(out_llama_cpp_return_code))
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_POSITION_REJECTED => {
            refused_validated_entry("refused the position of an entry it had just added")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OUTPUT_REJECTED => {
            refused_validated_entry("refused the output flag of an entry it had just added")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_DECISION_ORDER_REJECTED => {
            refused_validated_entry("refused the decision order of an entry it had just added")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_EMPTY_SEQ_IDS => {
            refused_validated_entry("was given an entry without sequence ids")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_SEQ_IDS_OUT_OF_RANGE => {
            refused_validated_entry(
                "was given an entry whose sequence ids lie outside the sequence id buffer",
            )
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_ENTRIES_ARG => {
            refused_validated_entry("was given a null entries argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_BATCH_ARG => {
            refused_validated_entry("was given a null batch argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_SEQ_IDS_ARG => {
            refused_validated_entry("was given a null seq_ids argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG => {
            refused_validated_entry("was given a null out_llama_cpp_return_code argument")
        }
        llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_OUT_ERROR_ARG => {
            refused_validated_entry("was given a null out_error argument")
        }
        other => Err(OPERATION.unknown_status(i64::from(other))
        .into()),
    }
}

/// Tokens to decode together. Each entry names its sequences by a range of `seq_ids`, the layout
/// llama.cpp receives the whole batch in, in one call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaBatch {
    limits: ContextLimits,
    entries: Vec<llama_rs_batch_entry>,
    seq_ids: Vec<llama_seq_id>,
    output_token_indices: Vec<i32>,
    n_tokens: i32,
}

impl LlamaBatch {
    #[must_use]
    pub const fn new(limits: ContextLimits) -> Self {
        Self {
            limits,
            entries: Vec::new(),
            seq_ids: Vec::new(),
            output_token_indices: Vec::new(),
            n_tokens: 0,
        }
    }

    /// # Errors
    ///
    /// Returns [`ContextError`] when the batch already holds as many tokens as one decode of the
    /// context takes, the token is outside the vocabulary, `seq_ids` is empty or names a sequence
    /// the context does not have, or the token count would overflow.
    pub fn add(
        &mut self,
        sampled_token: SampledToken,
        pos: llama_pos,
        seq_ids: &[llama_seq_id],
        logits: bool,
    ) -> Result<(), ContextError> {
        if self.n_tokens.unsigned_abs() >= self.limits.batch {
            return Err(ContextError::BatchFull {
                n_batch: self.limits.batch,
            });
        }

        let token = sampled_token.token();

        if !token.is_within_vocabulary(self.limits.vocabulary_size) {
            return Err(ContextError::TokenOutsideVocabulary {
                token,
                n_vocab: self.limits.vocabulary_size,
            });
        }

        if seq_ids.is_empty() {
            return Err(ContextError::NoSequenceIds);
        }

        if let Some(seq_id) = seq_ids
            .iter()
            .copied()
            .find(|seq_id| !self.limits.holds_sequence(*seq_id))
        {
            return Err(ContextError::SequenceIdOutOfRange {
                seq_id,
                n_seq_max: self.limits.max_sequences,
            });
        }

        let token_index = self.n_tokens;
        self.n_tokens = token_index
            .checked_add(1)
            .ok_or(ContextError::TokenCountOverflow {
                n_tokens: token_index,
            })?;

        if logits {
            self.output_token_indices.push(token_index);
        }

        self.entries.push(llama_rs_batch_entry {
            token: token.0,
            pos,
            seq_ids_offset: self.seq_ids.len(),
            seq_ids_count: seq_ids.len(),
            output: logits,
            decision_order: DecisionOrder::None.to_raw(),
        });
        self.seq_ids.extend_from_slice(seq_ids);

        Ok(())
    }

    /// Adds `tokens` at positions counting from zero in `seq_id`, with logits for the last one,
    /// or for every one when `logits_all` is set.
    ///
    /// # Errors
    ///
    /// Returns the [`ContextError`] of the first token [`Self::add`] refuses.
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: llama_seq_id,
        logits_all: bool,
    ) -> Result<(), ContextError> {
        let last_token_index = tokens.len().checked_sub(1);

        for ((token_index, token), pos) in tokens.iter().enumerate().zip(0..) {
            self.add(
                SampledToken::Content(*token),
                pos,
                &[seq_id],
                logits_all || Some(token_index) == last_token_index,
            )?;
        }

        Ok(())
    }

    /// Hands every token of this batch to llama.cpp's `batch_ext` in one call, replacing the
    /// tokens it held.
    ///
    /// # Safety
    ///
    /// `batch_ext` must point to a batch that llama.cpp initialized and has not freed.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError`] when llama.cpp refuses an entry or the wrapper fails.
    pub unsafe fn assign_to(
        &self,
        batch_ext: NonNull<llama_cpp_bindings_sys::llama_batch_ext>,
    ) -> Result<(), ContextError> {
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_batch_ext_assign_entries(
                batch_ext.as_ptr(),
                self.entries.as_ptr(),
                self.entries.len(),
                self.seq_ids.as_ptr(),
                self.seq_ids.len(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };

        batch_ext_assign_entries_status_to_result(status, out_llama_cpp_return_code, out_error)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.seq_ids.clear();
        self.output_token_indices.clear();
        self.n_tokens = 0;
    }

    #[must_use]
    pub const fn limits(&self) -> ContextLimits {
        self.limits
    }

    #[must_use]
    pub const fn n_tokens(&self) -> i32 {
        self.n_tokens
    }

    #[must_use]
    pub fn output_token_indices(&self) -> &[i32] {
        &self.output_token_indices
    }

    /// # Errors
    ///
    /// Returns [`ContextError::TokenIndexOutsideBatch`] when the batch holds no token at
    /// `token_index`.
    pub fn set_decision_order(
        &mut self,
        token_index: i32,
        decision_order: DecisionOrder,
    ) -> Result<(), ContextError> {
        let n_tokens = self.n_tokens;
        let entry = usize::try_from(token_index)
            .ok()
            .and_then(|entry_index| self.entries.get_mut(entry_index))
            .ok_or(ContextError::TokenIndexOutsideBatch {
                token_index,
                n_tokens,
            })?;

        entry.decision_order = decision_order.to_raw();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_sys::llama_rs_batch_entry;
    use llama_cpp_ffi_status::ffi_error::FfiError;
    use llama_cpp_model::llama_token::LlamaToken;
    use llama_cpp_model::sampled_token::SampledToken;

    use super::LlamaBatch;
    use super::batch_ext_assign_entries_status_to_result;
    use crate::context_error::ContextError;
    use crate::context_limits::ContextLimits;
    use crate::decision_order::DecisionOrder;

    const LIMITS: ContextLimits = ContextLimits {
        batch: 4,
        micro_batch: 4,
        max_sequences: 2,
        vocabulary_size: 100,
    };

    fn content(token_id: i32) -> SampledToken {
        SampledToken::Content(LlamaToken::new(token_id))
    }

    #[test]
    fn a_batch_whose_token_count_reached_its_type_limit_refuses_another_token() {
        let mut batch = LlamaBatch {
            n_tokens: i32::MAX,
            ..LlamaBatch::new(ContextLimits {
                batch: u32::MAX,
                ..LIMITS
            })
        };

        assert_eq!(
            batch.add(content(1), 0, &[0], false),
            Err(ContextError::TokenCountOverflow { n_tokens: i32::MAX })
        );
    }

    #[test]
    fn added_tokens_are_kept_in_order_with_their_outputs() {
        let mut batch = LlamaBatch::new(LIMITS);

        batch.add(content(1), 0, &[0], false).unwrap();
        batch.add(content(2), 1, &[0, 1], true).unwrap();

        assert_eq!(
            batch.entries,
            [
                llama_rs_batch_entry {
                    token: 1,
                    pos: 0,
                    seq_ids_offset: 0,
                    seq_ids_count: 1,
                    output: false,
                    decision_order: llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_NONE,
                },
                llama_rs_batch_entry {
                    token: 2,
                    pos: 1,
                    seq_ids_offset: 1,
                    seq_ids_count: 2,
                    output: true,
                    decision_order: llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_NONE,
                },
            ]
        );
        assert_eq!(batch.seq_ids, [0, 0, 1]);
        assert_eq!(batch.output_token_indices(), [1]);
        assert_eq!(batch.n_tokens(), 2);
    }

    #[test]
    fn a_decision_order_is_kept_on_the_token_it_marks_only() {
        let mut batch = LlamaBatch::new(LIMITS);
        batch
            .add_sequence(&[1, 2].map(LlamaToken::new), 0, false)
            .unwrap();

        batch.set_decision_order(1, DecisionOrder::Option).unwrap();

        assert_eq!(
            batch
                .entries
                .iter()
                .map(|entry| entry.decision_order)
                .collect::<Vec<_>>(),
            [
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_NONE,
                llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_OPTION,
            ]
        );
    }

    #[test]
    fn a_decision_order_for_a_token_the_batch_does_not_hold_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);
        batch.add(content(1), 0, &[0], false).unwrap();

        assert_eq!(
            [-1, 1]
                .map(|token_index| batch
                    .set_decision_order(token_index, DecisionOrder::QuestionChoice)),
            [-1, 1].map(|token_index| Err(ContextError::TokenIndexOutsideBatch {
                token_index,
                n_tokens: 1,
            }))
        );
    }

    #[test]
    fn clearing_forgets_every_token_and_output() {
        let mut batch = LlamaBatch::new(LIMITS);
        batch.add(content(1), 0, &[0], true).unwrap();

        batch.clear();

        assert_eq!(batch, LlamaBatch::new(LIMITS));
    }

    #[test]
    fn a_batch_holding_one_decode_of_tokens_refuses_another() {
        let mut batch = LlamaBatch::new(LIMITS);
        batch
            .add_sequence(&[1, 2, 3, 4].map(LlamaToken::new), 0, false)
            .unwrap();

        assert_eq!(
            batch.add(content(5), 4, &[0], false),
            Err(ContextError::BatchFull { n_batch: 4 })
        );
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(content(100), 0, &[0], false),
            Err(ContextError::TokenOutsideVocabulary {
                token: LlamaToken::new(100),
                n_vocab: 100,
            })
        );
    }

    #[test]
    fn the_last_token_of_the_vocabulary_is_accepted() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(batch.add(content(99), 0, &[0], false), Ok(()));
    }

    #[test]
    fn a_negative_token_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(content(-1), 0, &[0], false),
            Err(ContextError::TokenOutsideVocabulary {
                token: LlamaToken::new(-1),
                n_vocab: 100,
            })
        );
    }

    #[test]
    fn a_token_in_a_negative_sequence_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(content(1), 0, &[-1], false),
            Err(ContextError::SequenceIdOutOfRange {
                seq_id: -1,
                n_seq_max: 2,
            })
        );
    }

    #[test]
    fn a_token_without_sequences_is_refused() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(content(1), 0, &[], false),
            Err(ContextError::NoSequenceIds)
        );
    }

    #[test]
    fn a_token_in_a_sequence_the_context_lacks_is_refused_and_not_added() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add(content(1), 0, &[0, 2], false),
            Err(ContextError::SequenceIdOutOfRange {
                seq_id: 2,
                n_seq_max: 2,
            })
        );
        assert_eq!(batch, LlamaBatch::new(LIMITS));
    }

    #[test]
    fn a_sequence_requests_logits_for_its_last_token_only() {
        let mut batch = LlamaBatch::new(LIMITS);

        batch
            .add_sequence(&[10, 20, 30].map(LlamaToken::new), 1, false)
            .unwrap();

        assert_eq!(batch.output_token_indices(), [2]);
        assert_eq!(
            batch.entries.last(),
            Some(&llama_rs_batch_entry {
                token: 30,
                pos: 2,
                seq_ids_offset: 2,
                seq_ids_count: 1,
                output: true,
                decision_order: llama_cpp_bindings_sys::LLAMA_RS_DECISION_ORDER_NONE,
            })
        );
    }

    #[test]
    fn a_sequence_with_logits_for_all_requests_every_token() {
        let mut batch = LlamaBatch::new(LIMITS);

        batch
            .add_sequence(&[10, 20, 30].map(LlamaToken::new), 0, true)
            .unwrap();

        assert_eq!(batch.output_token_indices(), [0, 1, 2]);
    }

    #[test]
    fn a_sequence_longer_than_one_decode_stops_at_the_first_refused_token() {
        let mut batch = LlamaBatch::new(LIMITS);

        assert_eq!(
            batch.add_sequence(&[1, 2, 3, 4, 5].map(LlamaToken::new), 0, false),
            Err(ContextError::BatchFull { n_batch: 4 })
        );
        assert_eq!(batch.n_tokens(), 4);
    }

    #[test]
    fn assigning_batch_entries_reports_each_refused_or_malformed_entry_and_null_argument_as_a_contract_violation()
     {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_BATCH_FULL,
                "refused an entry of a batch validated against its limits as full",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_INVALID_TOKEN,
                "refused a token validated against the vocabulary",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_INVALID_SEQUENCE_ID,
                "refused a sequence id validated against the context",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_POSITION_REJECTED,
                "refused the position of an entry it had just added",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OUTPUT_REJECTED,
                "refused the output flag of an entry it had just added",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_DECISION_ORDER_REJECTED,
                "refused the decision order of an entry it had just added",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_EMPTY_SEQ_IDS,
                "was given an entry without sequence ids",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_SEQ_IDS_OUT_OF_RANGE,
                "was given an entry whose sequence ids lie outside the sequence id buffer",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_ENTRIES_ARG,
                "was given a null entries argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_BATCH_ARG,
                "was given a null batch argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_SEQ_IDS_ARG,
                "was given a null seq_ids argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG,
                "was given a null out_llama_cpp_return_code argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                batch_ext_assign_entries_status_to_result(status, 0, ptr::null_mut()),
                Err(ContextError::Ffi(FfiError::ContractViolated {
                    operation: "llama_rs_batch_ext_assign_entries",
                    detail,
                }))
            );
        }
    }

    #[test]
    fn an_undocumented_code_llama_cpp_returns_for_an_entry_is_reported_with_its_value() {
        assert_eq!(
            batch_ext_assign_entries_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_RETURNED_UNKNOWN_CODE,
                -7,
                ptr::null_mut(),
            ),
            Err(ContextError::Ffi(FfiError::UnknownStatus {
                operation: "llama_batch_ext_add_token",
                code: -7,
            }))
        );
    }

    #[test]
    fn adding_a_batch_entry_reports_memory_exhaustion() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_ERROR_STRING_ALLOCATION_FAILED,
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_OUT_OF_MEMORY,
            ]
            .map(|status| batch_ext_assign_entries_status_to_result(status, 0, ptr::null_mut())),
            [
                Err(ContextError::BatchTransferNotEnoughMemory),
                Err(ContextError::BatchTransferLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn adding_a_batch_entry_reports_an_exception_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"entry insertion threw".as_ptr())
        };

        assert_eq!(
            batch_ext_assign_entries_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            ),
            Err(ContextError::BatchTransferReported {
                message: "entry insertion threw".to_owned()
            })
        );
    }

    #[test]
    fn adding_a_batch_entry_reports_an_exception_without_a_message_as_a_contract_violation() {
        assert_eq!(
            batch_ext_assign_entries_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
            ),
            Err(ContextError::Ffi(FfiError::ContractViolated {
                    operation: "llama_rs_batch_ext_assign_entries",
                    detail: "reported a thrown C++ exception without an error message",
                }))
        );
    }

    #[test]
    fn adding_a_batch_entry_preserves_an_unknown_status() {
        assert_eq!(
            batch_ext_assign_entries_status_to_result(255, 0, ptr::null_mut()),
            Err(ContextError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_batch_ext_assign_entries",
                code: 255,
            }))
        );
    }
}
