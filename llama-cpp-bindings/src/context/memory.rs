use std::ffi::c_char;
use std::ffi::c_int;
use std::num::NonZeroU8;
use std::ptr;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::context::LlamaContext;
use crate::context::memory_of;
use crate::error::memory_seq_add_error::MemorySeqAddError;
use crate::error::memory_seq_cp_error::MemorySeqCpError;
use crate::error::memory_seq_div_error::MemorySeqDivError;
use crate::error::memory_seq_pos_max_error::MemorySeqPosMaxError;
use crate::error::memory_seq_rm_error::MemorySeqRmError;
use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;
use crate::error::unknown_rope_type::UnknownRopeType;
use crate::model::rope_type::RopeType;

fn memory_seq_add_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_add_status,
    out_error: *mut c_char,
) -> Result<(), MemorySeqAddError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM => {
            Err(MemorySeqAddError::MemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MemorySeqAddError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MemorySeqAddError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_memory_seq_add",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MemorySeqAddError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG => Err(FfiContractError {
            operation: "llama_rs_memory_seq_add",
            detail: "was given a null ctx argument",
        }
        .into()),
        other => Err(FfiStatusError {
            operation: "llama_rs_memory_seq_add",
            code: i64::from(other),
        }
        .into()),
    }
}

fn memory_seq_div_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_div_status,
    out_error: *mut c_char,
) -> Result<(), MemorySeqDivError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM => {
            Err(MemorySeqDivError::MemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MemorySeqDivError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MemorySeqDivError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_memory_seq_div",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MemorySeqDivError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG => Err(FfiContractError {
            operation: "llama_rs_memory_seq_div",
            detail: "was given a null ctx argument",
        }
        .into()),
        other => Err(FfiStatusError {
            operation: "llama_rs_memory_seq_div",
            code: i64::from(other),
        }
        .into()),
    }
}

fn memory_seq_pos_max_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_pos_max_status,
    position: i32,
    out_error: *mut c_char,
) -> Result<i32, MemorySeqPosMaxError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_OK => Ok(position),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_CTX_ARG => Err(FfiContractError {
            operation: "llama_rs_memory_seq_pos_max",
            detail: "context pointer was null",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_POSITION_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "output position pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_ERROR_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "output error pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_MEM => {
            Err(MemorySeqPosMaxError::MemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MemorySeqPosMaxError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MemorySeqPosMaxError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_memory_seq_pos_max",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MemorySeqPosMaxError::Reported { message })
        }
        other => Err(FfiStatusError {
            operation: "llama_rs_memory_seq_pos_max",
            code: i64::from(other),
        }
        .into()),
    }
}

impl LlamaContext<'_> {
    #[must_use]
    pub fn has_memory(&self) -> bool {
        memory_of(self.context).is_some()
    }

    fn shifts_positions_one_per_embedding(&self) -> Result<bool, UnknownRopeType> {
        self.model
            .rope_type()
            .map(RopeType::has_one_position_per_embedding)
    }

    fn required_memory_handle(
        &self,
    ) -> Result<llama_cpp_bindings_sys::llama_memory_t, MemorySeqCpError> {
        memory_of(self.context).ok_or(MemorySeqCpError::MemoryHandleUnavailable)
    }

    /// # Errors
    /// If either sequence id names no sequence of the context, either position exceeds
    /// [`i32::MAX`], or the context has no memory module.
    pub fn memory_seq_cp(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), MemorySeqCpError> {
        self.limits.ensure_sequence_id(src)?;
        self.limits.ensure_sequence_id(dest)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqCpError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqCpError::P1TooLarge)?;
        let mem = self.required_memory_handle()?;
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_cp(mem, src, dest, p0, p1) };

        Ok(())
    }

    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`], the sequence id names no
    /// sequence of the context, or llama.cpp reports that the partial sequence could not be
    /// removed. `None` removes from every sequence. A context without a memory module holds no
    /// KV cache, so the removal trivially succeeds.
    pub fn memory_seq_rm(
        &mut self,
        seq_id: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), MemorySeqRmError> {
        let seq_id = seq_id
            .map(i32::try_from)
            .transpose()
            .map_err(MemorySeqRmError::SeqIdTooLarge)?;

        if let Some(sequence) = seq_id {
            self.limits.ensure_sequence_id(sequence)?;
        }

        let seq_id = seq_id.unwrap_or(-1);
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqRmError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqRmError::P1TooLarge)?;
        let Some(mem) = memory_of(self.context) else {
            return Ok(());
        };

        if unsafe { llama_cpp_bindings_sys::llama_memory_seq_rm(mem, seq_id, p0, p1) } {
            return Ok(());
        }

        Err(MemorySeqRmError::PartialSequenceNotRemoved { seq_id, p0, p1 })
    }

    pub fn memory_clear(&mut self) {
        let Some(mem) = memory_of(self.context) else {
            return;
        };
        let clear_data_buffers = true;
        unsafe { llama_cpp_bindings_sys::llama_memory_clear(mem, clear_data_buffers) };
    }

    /// # Errors
    /// If the sequence id names no sequence of the context.
    pub fn memory_seq_keep(&mut self, seq_id: i32) -> Result<(), SequenceIdOutOfRange> {
        self.limits.ensure_sequence_id(seq_id)?;

        if let Some(mem) = memory_of(self.context) {
            unsafe { llama_cpp_bindings_sys::llama_memory_seq_keep(mem, seq_id) };
        }

        Ok(())
    }

    /// # Errors
    /// If the sequence id names no sequence of the context, either position exceeds
    /// [`i32::MAX`], or the underlying memory operation reports a failure.
    pub fn memory_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), MemorySeqAddError> {
        self.limits.ensure_sequence_id(seq_id)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqAddError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqAddError::P1TooLarge)?;

        if !self.shifts_positions_one_per_embedding()? {
            return Err(MemorySeqAddError::IncompatibleRopeType);
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

    /// # Errors
    /// If the sequence id names no sequence of the context, either position exceeds
    /// [`i32::MAX`], or the underlying memory operation reports a failure.
    pub fn memory_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        divisor: NonZeroU8,
    ) -> Result<(), MemorySeqDivError> {
        self.limits.ensure_sequence_id(seq_id)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqDivError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(MemorySeqDivError::P1TooLarge)?;

        if !self.shifts_positions_one_per_embedding()? {
            return Err(MemorySeqDivError::IncompatibleRopeType);
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

    /// # Errors
    ///
    /// Returns [`MemorySeqPosMaxError`] if the sequence does not exist or the memory lookup fails.
    pub fn memory_seq_pos_max(&self, seq_id: i32) -> Result<i32, MemorySeqPosMaxError> {
        self.limits.ensure_sequence_id(seq_id)?;

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
}

#[cfg(test)]
mod memory_seq_add_status_to_result_tests {
    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::memory_seq_add_status_to_result;
    use crate::error::memory_seq_add_error::MemorySeqAddError;

    #[test]
    fn add_ok_status_maps_to_ok() {
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
            Err(MemorySeqAddError::MemoryHandleUnavailable)
        );
    }

    #[test]
    fn add_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(MemorySeqAddError::NotEnoughMemory)
        );
    }

    #[test]
    fn add_llama_cpp_exception_status_without_a_message_is_a_contract_violation() {
        assert_eq!(
            memory_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_memory_seq_add",
                detail: "reported a thrown C++ exception without an error message",
            })
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
            Err(MemorySeqAddError::Reported {
                message: "shift threw".to_owned()
            })
        );
    }

    #[test]
    fn add_unknown_status_is_preserved() {
        let result = memory_seq_add_status_to_result(255, ptr::null_mut());

        assert_eq!(
            result,
            Err(MemorySeqAddError::FfiStatus(FfiStatusError {
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
                FfiContractError {
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
        assert_eq!(outcome.err(), Some(MemorySeqAddError::LlamaCppOutOfMemory));
    }
}

#[cfg(test)]
mod memory_seq_div_status_to_result_tests {
    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::memory_seq_div_status_to_result;
    use crate::error::memory_seq_div_error::MemorySeqDivError;

    #[test]
    fn div_ok_status_maps_to_ok() {
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
            Err(MemorySeqDivError::MemoryHandleUnavailable)
        );
    }

    #[test]
    fn div_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(MemorySeqDivError::NotEnoughMemory)
        );
    }

    #[test]
    fn div_llama_cpp_exception_status_without_a_message_is_a_contract_violation() {
        assert_eq!(
            memory_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_memory_seq_div",
                detail: "reported a thrown C++ exception without an error message",
            })
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
            Err(MemorySeqDivError::Reported {
                message: "division threw".to_owned()
            })
        );
    }

    #[test]
    fn div_unknown_status_is_preserved() {
        let result = memory_seq_div_status_to_result(255, ptr::null_mut());

        assert_eq!(
            result,
            Err(MemorySeqDivError::FfiStatus(FfiStatusError {
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
                FfiContractError {
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
        assert_eq!(outcome.err(), Some(MemorySeqDivError::LlamaCppOutOfMemory));
    }
}

#[cfg(test)]
mod memory_seq_pos_max_status_to_result_tests {
    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::memory_seq_pos_max_status_to_result;
    use crate::error::memory_seq_pos_max_error::MemorySeqPosMaxError;

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
            Err(MemorySeqPosMaxError::FfiContract(FfiContractError {
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
            Err(MemorySeqPosMaxError::FfiContract(FfiContractError {
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
            Err(MemorySeqPosMaxError::FfiContract(FfiContractError {
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
            Err(MemorySeqPosMaxError::MemoryHandleUnavailable)
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
            Err(MemorySeqPosMaxError::NotEnoughMemory)
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
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "reported a thrown C++ exception without an error message",
            })
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
            Err(MemorySeqPosMaxError::Reported {
                message: "position lookup threw".to_owned()
            })
        );
    }

    #[test]
    fn seq_pos_max_unknown_status_is_preserved() {
        assert_eq!(
            memory_seq_pos_max_status_to_result(255, -1, ptr::null_mut(),),
            Err(MemorySeqPosMaxError::FfiStatus(FfiStatusError {
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
            Some(MemorySeqPosMaxError::LlamaCppOutOfMemory)
        );
    }
}
