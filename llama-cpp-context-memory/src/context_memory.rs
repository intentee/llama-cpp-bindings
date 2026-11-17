use std::ffi::c_char;
use std::ffi::c_int;
use std::num::NonZeroU8;
use std::num::TryFromIntError;
use std::ptr;

use llama_cpp_context::context_limits::ContextLimits;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_ffi_status::wrapper_operation::WrapperOperation;

use crate::context_memory_error::ContextMemoryError;

fn ensure_sequence_id(
    limits: ContextLimits,
    seq_id: i32,
    out_of_range: impl FnOnce(u32) -> ContextMemoryError,
) -> Result<(), ContextMemoryError> {
    if limits.holds_sequence(seq_id) {
        Ok(())
    } else {
        Err(out_of_range(limits.max_sequences))
    }
}

fn open_ended_position(position: Option<u32>) -> Result<i32, TryFromIntError> {
    position.map_or(Ok(-1), i32::try_from)
}

fn memory_seq_add_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_add_status,
    out_error: *mut c_char,
) -> Result<(), ContextMemoryError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_memory_seq_add");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM => {
            Err(ContextMemoryError::SeqAddMemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextMemoryError::SeqAddNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextMemoryError::SeqAddLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { OPERATION.read_and_free_exception_message(out_error) }?;
            Err(ContextMemoryError::SeqAddReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG => Err(OPERATION
            .contract_violated("was given a null ctx argument")
            .into()),
        other => Err(OPERATION.unknown_status(i64::from(other)).into()),
    }
}

fn memory_seq_div_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_div_status,
    out_error: *mut c_char,
) -> Result<(), ContextMemoryError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_memory_seq_div");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM => {
            Err(ContextMemoryError::SeqDivMemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextMemoryError::SeqDivNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextMemoryError::SeqDivLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { OPERATION.read_and_free_exception_message(out_error) }?;
            Err(ContextMemoryError::SeqDivReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG => Err(OPERATION
            .contract_violated("was given a null ctx argument")
            .into()),
        other => Err(OPERATION.unknown_status(i64::from(other)).into()),
    }
}

fn memory_seq_pos_max_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_pos_max_status,
    position: i32,
    out_error: *mut c_char,
) -> Result<i32, ContextMemoryError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_memory_seq_pos_max");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_OK => Ok(position),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_CTX_ARG => Err(OPERATION
            .contract_violated("context pointer was null")
            .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_POSITION_ARG => Err(OPERATION
            .contract_violated("output position pointer was null")
            .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_ERROR_ARG => Err(OPERATION
            .contract_violated("output error pointer was null")
            .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_MEM => {
            Err(ContextMemoryError::SeqPosMaxMemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextMemoryError::SeqPosMaxNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextMemoryError::SeqPosMaxLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { OPERATION.read_and_free_exception_message(out_error) }?;
            Err(ContextMemoryError::SeqPosMaxReported { message })
        }
        other => Err(OPERATION.unknown_status(i64::from(other)).into()),
    }
}

pub trait ContextMemory {
    #[must_use]
    fn has_memory(&self) -> bool;

    fn memory_clear(&self);

    /// # Errors
    /// If the sequence id names no sequence of the context, either position exceeds
    /// [`i32::MAX`], or the underlying memory operation reports a failure.
    fn memory_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), ContextMemoryError>;

    /// # Errors
    /// If either sequence id names no sequence of the context, either position exceeds
    /// [`i32::MAX`], or the context has no memory module.
    fn memory_seq_cp(
        &self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), ContextMemoryError>;

    /// # Errors
    /// If the sequence id names no sequence of the context, either position exceeds
    /// [`i32::MAX`], or the underlying memory operation reports a failure.
    fn memory_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        divisor: NonZeroU8,
    ) -> Result<(), ContextMemoryError>;

    /// # Errors
    /// If the sequence id names no sequence of the context.
    fn memory_seq_keep(&self, seq_id: i32) -> Result<(), ContextMemoryError>;

    /// # Errors
    ///
    /// Returns [`ContextMemoryError`] if the sequence does not exist or the memory lookup fails.
    fn memory_seq_pos_max(&self, seq_id: i32) -> Result<i32, ContextMemoryError>;

    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`], the sequence id names no
    /// sequence of the context, or llama.cpp reports that the partial sequence could not be
    /// removed. `None` removes from every sequence. A context without a memory module holds no
    /// KV cache, so the removal trivially succeeds.
    fn memory_seq_rm(
        &self,
        seq_id: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), ContextMemoryError>;
}

impl ContextMemory for LlamaContext<'_> {
    fn has_memory(&self) -> bool {
        self.memory().is_some()
    }

    fn memory_clear(&self) {
        let Some(mem) = self.memory() else {
            return;
        };
        let clear_data_buffers = true;
        unsafe { llama_cpp_bindings_sys::llama_memory_clear(mem, clear_data_buffers) };
    }

    fn memory_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), ContextMemoryError> {
        ensure_sequence_id(self.limits(), seq_id, |n_seq_max| {
            ContextMemoryError::SeqAddSequenceIdOutOfRange { seq_id, n_seq_max }
        })?;
        let p0 =
            open_ended_position(p0).map_err(ContextMemoryError::SeqAddStartPositionTooLarge)?;
        let p1 = open_ended_position(p1).map_err(ContextMemoryError::SeqAddEndPositionTooLarge)?;

        let rope_type = self
            .model
            .rope_type()
            .map_err(ContextMemoryError::SeqAddUnknownRopeType)?;

        if !rope_type.has_one_position_per_embedding() {
            return Err(ContextMemoryError::SeqAddIncompatibleRopeType);
        }

        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_add(
                self.context.as_ptr().cast_const(),
                seq_id,
                p0,
                p1,
                delta,
                &raw mut out_error,
            )
        };
        memory_seq_add_status_to_result(status, out_error)
    }

    fn memory_seq_cp(
        &self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), ContextMemoryError> {
        ensure_sequence_id(self.limits(), src, |n_seq_max| {
            ContextMemoryError::SeqCpSequenceIdOutOfRange {
                seq_id: src,
                n_seq_max,
            }
        })?;
        ensure_sequence_id(self.limits(), dest, |n_seq_max| {
            ContextMemoryError::SeqCpSequenceIdOutOfRange {
                seq_id: dest,
                n_seq_max,
            }
        })?;
        let p0 = open_ended_position(p0).map_err(ContextMemoryError::SeqCpStartPositionTooLarge)?;
        let p1 = open_ended_position(p1).map_err(ContextMemoryError::SeqCpEndPositionTooLarge)?;
        let mem = self
            .memory()
            .ok_or(ContextMemoryError::SeqCpMemoryHandleUnavailable)?;
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_cp(mem, src, dest, p0, p1) };

        Ok(())
    }

    fn memory_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        divisor: NonZeroU8,
    ) -> Result<(), ContextMemoryError> {
        ensure_sequence_id(self.limits(), seq_id, |n_seq_max| {
            ContextMemoryError::SeqDivSequenceIdOutOfRange { seq_id, n_seq_max }
        })?;
        let p0 =
            open_ended_position(p0).map_err(ContextMemoryError::SeqDivStartPositionTooLarge)?;
        let p1 = open_ended_position(p1).map_err(ContextMemoryError::SeqDivEndPositionTooLarge)?;

        let rope_type = self
            .model
            .rope_type()
            .map_err(ContextMemoryError::SeqDivUnknownRopeType)?;

        if !rope_type.has_one_position_per_embedding() {
            return Err(ContextMemoryError::SeqDivIncompatibleRopeType);
        }

        let divisor = c_int::from(divisor.get());
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_div(
                self.context.as_ptr().cast_const(),
                seq_id,
                p0,
                p1,
                divisor,
                &raw mut out_error,
            )
        };
        memory_seq_div_status_to_result(status, out_error)
    }

    fn memory_seq_keep(&self, seq_id: i32) -> Result<(), ContextMemoryError> {
        ensure_sequence_id(self.limits(), seq_id, |n_seq_max| {
            ContextMemoryError::SeqKeepSequenceIdOutOfRange { seq_id, n_seq_max }
        })?;

        if let Some(mem) = self.memory() {
            unsafe { llama_cpp_bindings_sys::llama_memory_seq_keep(mem, seq_id) };
        }

        Ok(())
    }

    fn memory_seq_pos_max(&self, seq_id: i32) -> Result<i32, ContextMemoryError> {
        ensure_sequence_id(self.limits(), seq_id, |n_seq_max| {
            ContextMemoryError::SeqPosMaxSequenceIdOutOfRange { seq_id, n_seq_max }
        })?;

        let mut position = -1;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_memory_seq_pos_max(
                self.context.as_ptr().cast_const(),
                seq_id,
                &raw mut position,
                &raw mut out_error,
            )
        };

        memory_seq_pos_max_status_to_result(status, position, out_error)
    }

    fn memory_seq_rm(
        &self,
        seq_id: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), ContextMemoryError> {
        let seq_id = seq_id
            .map(i32::try_from)
            .transpose()
            .map_err(ContextMemoryError::SeqRmSequenceIdTooLarge)?;

        if let Some(sequence) = seq_id {
            ensure_sequence_id(self.limits(), sequence, |n_seq_max| {
                ContextMemoryError::SeqRmSequenceIdOutOfRange {
                    seq_id: sequence,
                    n_seq_max,
                }
            })?;
        }

        let seq_id = seq_id.unwrap_or(-1);
        let p0 = open_ended_position(p0).map_err(ContextMemoryError::SeqRmStartPositionTooLarge)?;
        let p1 = open_ended_position(p1).map_err(ContextMemoryError::SeqRmEndPositionTooLarge)?;
        let Some(mem) = self.memory() else {
            return Ok(());
        };

        if unsafe { llama_cpp_bindings_sys::llama_memory_seq_rm(mem, seq_id, p0, p1) } {
            return Ok(());
        }

        Err(ContextMemoryError::SeqRmPartialSequenceNotRemoved { seq_id, p0, p1 })
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_context::context_limits::ContextLimits;
    use llama_cpp_ffi_status::ffi_error::FfiError;

    use super::ensure_sequence_id;
    use super::memory_seq_add_status_to_result;
    use super::memory_seq_div_status_to_result;
    use super::memory_seq_pos_max_status_to_result;
    use super::open_ended_position;
    use crate::context_memory_error::ContextMemoryError;

    const LIMITS: ContextLimits = ContextLimits {
        batch: 512,
        micro_batch: 128,
        max_sequences: 2,
        vocabulary_size: 100,
    };

    fn keep_out_of_range(seq_id: i32) -> impl FnOnce(u32) -> ContextMemoryError {
        move |n_seq_max| ContextMemoryError::SeqKeepSequenceIdOutOfRange { seq_id, n_seq_max }
    }

    #[test]
    fn a_sequence_the_context_holds_is_accepted() {
        assert_eq!(ensure_sequence_id(LIMITS, 1, keep_out_of_range(1)), Ok(()));
    }

    #[test]
    fn a_sequence_the_context_lacks_is_refused_with_the_sequence_count() {
        assert_eq!(
            [-1, 2].map(|seq_id| ensure_sequence_id(LIMITS, seq_id, keep_out_of_range(seq_id))),
            [-1, 2].map(
                |seq_id| Err(ContextMemoryError::SeqKeepSequenceIdOutOfRange {
                    seq_id,
                    n_seq_max: 2,
                })
            )
        );
    }

    #[test]
    fn an_absent_position_reads_as_the_open_end_llama_cpp_marks_with_minus_one() {
        assert_eq!(open_ended_position(None), Ok(-1));
    }

    #[test]
    fn a_position_past_i32_is_refused_with_the_conversion_failure() {
        assert_eq!(
            open_ended_position(Some(u32::MAX)),
            Err(i32::try_from(u32::MAX).unwrap_err())
        );
    }

    #[test]
    fn a_successful_position_shift_status_is_ok() {
        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_OK,
                ptr::null_mut(),
            ),
            Ok(())
        );
    }

    #[test]
    fn add_null_mem_status_maps_to_memory_handle_unavailable() {
        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::SeqAddMemoryHandleUnavailable)
        );
    }

    #[test]
    fn add_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::SeqAddNotEnoughMemory)
        );
    }

    #[test]
    fn add_llama_cpp_exception_status_without_a_message_is_a_contract_violation() {
        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(FfiError::ContractViolated {
                operation: "llama_rs_memory_seq_add",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn add_llama_cpp_exception_status_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"shift threw".as_ptr()) };

        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION,
                out_error,
            ),
            Err(ContextMemoryError::SeqAddReported {
                message: "shift threw".to_owned()
            })
        );
    }

    #[test]
    fn add_unknown_status_is_preserved() {
        let result = memory_seq_add_status_to_result(255, ptr::null_mut());

        assert_eq!(
            result,
            Err(ContextMemoryError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_memory_seq_add",
                code: 255,
            }))
        );
    }

    #[test]
    fn memory_seq_add_status_to_result_reports_a_null_ctx_arg_as_a_contract_violation() {
        let outcome = memory_seq_add_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_memory_seq_add",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn memory_seq_add_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = memory_seq_add_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(ContextMemoryError::SeqAddLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn a_successful_position_division_status_is_ok() {
        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_OK,
                ptr::null_mut(),
            ),
            Ok(())
        );
    }

    #[test]
    fn div_null_mem_status_maps_to_memory_handle_unavailable() {
        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::SeqDivMemoryHandleUnavailable)
        );
    }

    #[test]
    fn div_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::SeqDivNotEnoughMemory)
        );
    }

    #[test]
    fn div_llama_cpp_exception_status_without_a_message_is_a_contract_violation() {
        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(FfiError::ContractViolated {
                operation: "llama_rs_memory_seq_div",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn div_llama_cpp_exception_status_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"division threw".as_ptr()) };

        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_THREW_CXX_EXCEPTION,
                out_error,
            ),
            Err(ContextMemoryError::SeqDivReported {
                message: "division threw".to_owned()
            })
        );
    }

    #[test]
    fn div_unknown_status_is_preserved() {
        let result = memory_seq_div_status_to_result(255, ptr::null_mut());

        assert_eq!(
            result,
            Err(ContextMemoryError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_memory_seq_div",
                code: 255,
            }))
        );
    }

    #[test]
    fn memory_seq_div_status_to_result_reports_a_null_ctx_arg_as_a_contract_violation() {
        let outcome = memory_seq_div_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_memory_seq_div",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn memory_seq_div_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = memory_seq_div_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(ContextMemoryError::SeqDivLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn seq_pos_max_ok_status_returns_position() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_OK,
                17,
                ptr::null_mut(),
            ),
            Ok(17)
        );
    }

    #[test]
    fn seq_pos_max_null_context_status_is_contract_error() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_CTX_ARG,
                -1,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::Ffi(FfiError::ContractViolated {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "context pointer was null",
            }))
        );
    }

    #[test]
    fn seq_pos_max_null_output_position_status_is_contract_error() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_POSITION_ARG,
                -1,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::Ffi(FfiError::ContractViolated {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "output position pointer was null",
            }))
        );
    }

    #[test]
    fn seq_pos_max_null_output_error_status_is_contract_error() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_ERROR_ARG,
                -1,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::Ffi(FfiError::ContractViolated {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "output error pointer was null",
            }))
        );
    }

    #[test]
    fn seq_pos_max_null_memory_status_maps_to_memory_handle_unavailable() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_MEM,
                -1,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::SeqPosMaxMemoryHandleUnavailable)
        );
    }

    #[test]
    fn seq_pos_max_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_ERROR_STRING_ALLOCATION_FAILED,
                -1,
                ptr::null_mut(),
            ),
            Err(ContextMemoryError::SeqPosMaxNotEnoughMemory)
        );
    }

    #[test]
    fn seq_pos_max_llama_cpp_exception_status_without_a_message_is_a_contract_violation() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_THREW_CXX_EXCEPTION,
                -1,
                ptr::null_mut(),
            ),
            Err(FfiError::ContractViolated {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn seq_pos_max_llama_cpp_exception_status_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"position lookup threw".as_ptr())
        };

        assert_eq!(
            memory_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_THREW_CXX_EXCEPTION,
                -1,
                out_error,
            ),
            Err(ContextMemoryError::SeqPosMaxReported {
                message: "position lookup threw".to_owned()
            })
        );
    }

    #[test]
    fn seq_pos_max_unknown_status_is_preserved() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(255, -1, ptr::null_mut(),),
            Err(ContextMemoryError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_memory_seq_pos_max",
                code: 255,
            }))
        );
    }

    #[test]
    fn memory_seq_pos_max_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = memory_seq_pos_max_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(ContextMemoryError::SeqPosMaxLlamaCppOutOfMemory)
        );
    }
}
