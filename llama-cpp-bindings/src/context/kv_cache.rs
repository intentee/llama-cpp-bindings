use std::ffi::c_int;
use std::num::NonZeroU8;
use std::os::raw::c_char;
use std::ptr;

use crate::context::LlamaContext;
use crate::error::kv_cache_conversion_error::KvCacheConversionError;
use crate::error::{KvCacheSeqAddError, KvCacheSeqDivError, KvCacheSeqPosMaxError};
use llama_cpp_ffi_status::read_and_free_cpp_string;

fn kv_cache_seq_add_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_add_status,
    out_error: *mut c_char,
) -> Result<(), KvCacheSeqAddError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE => {
            Err(KvCacheSeqAddError::IncompatibleRopeType)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM => {
            Err(KvCacheSeqAddError::MemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED => {
            Err(KvCacheSeqAddError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_OUT_OF_MEMORY => {
            Err(KvCacheSeqAddError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_memory_seq_add",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(KvCacheSeqAddError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_add",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MODEL => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_add",
                detail: "was given a null model argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_memory_seq_add",
            code: i64::from(other),
        }
        .into()),
    }
}

fn kv_cache_seq_div_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_div_status,
    out_error: *mut c_char,
) -> Result<(), KvCacheSeqDivError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE => {
            Err(KvCacheSeqDivError::IncompatibleRopeType)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM => {
            Err(KvCacheSeqDivError::MemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED => {
            Err(KvCacheSeqDivError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_OUT_OF_MEMORY => {
            Err(KvCacheSeqDivError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_memory_seq_div",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(KvCacheSeqDivError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_div",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MODEL => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_div",
                detail: "was given a null model argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_memory_seq_div",
            code: i64::from(other),
        }
        .into()),
    }
}

fn kv_cache_seq_pos_max_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_memory_seq_pos_max_status,
    position: i32,
    seq_id: i32,
    out_error: *mut c_char,
) -> Result<i32, KvCacheSeqPosMaxError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_OK => Ok(position),
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "context pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_POSITION_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "output position pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "output error pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_MEM => {
            Err(KvCacheSeqPosMaxError::MemoryHandleUnavailable)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_SEQ_ID_OUT_OF_RANGE => {
            Err(KvCacheSeqPosMaxError::SequenceIdOutOfRange { seq_id })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_ERROR_STRING_ALLOCATION_FAILED => {
            Err(KvCacheSeqPosMaxError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_VENDORED_OUT_OF_MEMORY => {
            Err(KvCacheSeqPosMaxError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_memory_seq_pos_max",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(KvCacheSeqPosMaxError::Reported { message })
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_memory_seq_pos_max",
            code: i64::from(other),
        }
        .into()),
    }
}

impl LlamaContext<'_> {
    /// # Errors
    /// Returns [`KvCacheConversionError::MemoryHandleUnavailable`] when the context was
    /// built without a memory module, so a null handle is never handed to llama.cpp.
    fn memory_handle(
        &self,
    ) -> Result<llama_cpp_bindings_sys::llama_memory_t, KvCacheConversionError> {
        let mem = unsafe { llama_cpp_bindings_sys::llama_get_memory(self.context.as_ptr()) };

        if mem.is_null() {
            return Err(KvCacheConversionError::MemoryHandleUnavailable);
        }

        Ok(mem)
    }

    /// # Errors
    /// If either position exceeds [`i32::MAX`], or the context has no memory module.
    pub fn copy_kv_cache_seq(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = self.memory_handle()?;
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_cp(mem, src, dest, p0, p1) };

        Ok(())
    }

    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`], the context has no
    /// memory module, or llama.cpp reports that the partial sequence could not be removed.
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let src = src
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::SeqIdTooLarge)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = self.memory_handle()?;

        if unsafe { llama_cpp_bindings_sys::llama_memory_seq_rm(mem, src, p0, p1) } {
            return Ok(());
        }

        Err(KvCacheConversionError::PartialSequenceNotRemoved {
            seq_id: src,
            p0,
            p1,
        })
    }

    /// # Errors
    /// If the context has no memory module.
    pub fn clear_kv_cache(&mut self) -> Result<(), KvCacheConversionError> {
        let mem = self.memory_handle()?;
        let clear_data_buffers = true;
        unsafe { llama_cpp_bindings_sys::llama_memory_clear(mem, clear_data_buffers) };

        Ok(())
    }

    /// # Errors
    /// If the context has no memory module.
    pub fn kv_cache_seq_keep(&mut self, seq_id: i32) -> Result<(), KvCacheConversionError> {
        let mem = self.memory_handle()?;
        unsafe { llama_cpp_bindings_sys::llama_memory_seq_keep(mem, seq_id) };

        Ok(())
    }

    /// # Errors
    /// If either position exceeds [`i32::MAX`], or the underlying memory operation reports a failure.
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheSeqAddError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqAddError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqAddError::P1TooLarge)?;
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
        kv_cache_seq_add_status_to_result(status, out_error)
    }

    /// # Errors
    /// If either position exceeds [`i32::MAX`], or the underlying memory operation reports a failure.
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        divisor: NonZeroU8,
    ) -> Result<(), KvCacheSeqDivError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqDivError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheSeqDivError::P1TooLarge)?;
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
        kv_cache_seq_div_status_to_result(status, out_error)
    }

    /// # Errors
    ///
    /// Returns [`KvCacheSeqPosMaxError`] if the sequence does not exist or the memory lookup fails.
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> Result<i32, KvCacheSeqPosMaxError> {
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

        kv_cache_seq_pos_max_status_to_result(status, position, seq_id, out_error)
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::kv_cache_seq_add_status_to_result;
    use super::kv_cache_seq_div_status_to_result;
    use super::kv_cache_seq_pos_max_status_to_result;
    use crate::error::{KvCacheSeqAddError, KvCacheSeqDivError, KvCacheSeqPosMaxError};

    #[test]
    fn add_ok_status_maps_to_ok() {
        let result = kv_cache_seq_add_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_OK,
            ptr::null_mut(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn add_incompatible_rope_type_status_maps_to_incompatible_rope_type() {
        assert_eq!(
            kv_cache_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqAddError::IncompatibleRopeType)
        );
    }

    #[test]
    fn add_null_mem_status_maps_to_memory_handle_unavailable() {
        assert_eq!(
            kv_cache_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqAddError::MemoryHandleUnavailable)
        );
    }

    #[test]
    fn add_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            kv_cache_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqAddError::NotEnoughMemory)
        );
    }

    #[test]
    fn add_vendored_exception_status_without_a_message_is_a_contract_error_with_unknown_message() {
        assert_eq!(
            kv_cache_seq_add_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_add",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn add_unknown_status_is_preserved() {
        let result = kv_cache_seq_add_status_to_result(255, ptr::null_mut());

        assert_eq!(
            result,
            Err(KvCacheSeqAddError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_memory_seq_add",
                code: 255,
            }))
        );
    }

    #[test]
    fn div_ok_status_maps_to_ok() {
        let result = kv_cache_seq_div_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_OK,
            ptr::null_mut(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn div_incompatible_rope_type_status_maps_to_incompatible_rope_type() {
        assert_eq!(
            kv_cache_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqDivError::IncompatibleRopeType)
        );
    }

    #[test]
    fn div_null_mem_status_maps_to_memory_handle_unavailable() {
        assert_eq!(
            kv_cache_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqDivError::MemoryHandleUnavailable)
        );
    }

    #[test]
    fn div_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            kv_cache_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqDivError::NotEnoughMemory)
        );
    }

    #[test]
    fn div_vendored_exception_status_without_a_message_is_a_contract_error_with_unknown_message() {
        assert_eq!(
            kv_cache_seq_div_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_div",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn div_unknown_status_is_preserved() {
        let result = kv_cache_seq_div_status_to_result(255, ptr::null_mut());

        assert_eq!(
            result,
            Err(KvCacheSeqDivError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_memory_seq_div",
                code: 255,
            }))
        );
    }

    #[test]
    fn seq_pos_max_ok_status_returns_position() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_OK,
                17,
                2,
                ptr::null_mut(),
            ),
            Ok(17)
        );
    }

    #[test]
    fn seq_pos_max_null_context_status_is_contract_error() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_CTX_ARG,
                -1,
                2,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqPosMaxError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_memory_seq_pos_max",
                    detail: "context pointer was null",
                }
            ))
        );
    }

    #[test]
    fn seq_pos_max_null_output_position_status_is_contract_error() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_POSITION_ARG,
                -1,
                2,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqPosMaxError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_memory_seq_pos_max",
                    detail: "output position pointer was null",
                }
            ))
        );
    }

    #[test]
    fn seq_pos_max_null_output_error_status_is_contract_error() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_OUT_ERROR_ARG,
                -1,
                2,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqPosMaxError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_memory_seq_pos_max",
                    detail: "output error pointer was null",
                }
            ))
        );
    }

    #[test]
    fn seq_pos_max_null_memory_status_maps_to_memory_handle_unavailable() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_NULL_MEM,
                -1,
                2,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqPosMaxError::MemoryHandleUnavailable)
        );
    }

    #[test]
    fn seq_pos_max_out_of_range_status_preserves_sequence_id() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_SEQ_ID_OUT_OF_RANGE,
                -1,
                27,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqPosMaxError::SequenceIdOutOfRange { seq_id: 27 })
        );
    }

    #[test]
    fn seq_pos_max_allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_ERROR_STRING_ALLOCATION_FAILED,
                -1,
                2,
                ptr::null_mut(),
            ),
            Err(KvCacheSeqPosMaxError::NotEnoughMemory)
        );
    }

    #[test]
    fn seq_pos_max_vendored_exception_status_without_a_message_is_a_contract_error_error() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_VENDORED_THREW_CXX_EXCEPTION,
                -1,
                2,
                ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_memory_seq_pos_max",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn seq_pos_max_unknown_status_is_preserved() {
        assert_eq!(
            kv_cache_seq_pos_max_status_to_result(255, -1, 2, ptr::null_mut(),),
            Err(KvCacheSeqPosMaxError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_memory_seq_pos_max",
                code: 255,
            }))
        );
    }
}

#[cfg(test)]
mod ffi_contract_status_tests {
    use super::kv_cache_seq_add_status_to_result;
    use super::kv_cache_seq_div_status_to_result;
    use super::kv_cache_seq_pos_max_status_to_result;
    use crate::error::kv_cache_seq_add_error::KvCacheSeqAddError;
    use crate::error::kv_cache_seq_div_error::KvCacheSeqDivError;
    use crate::error::kv_cache_seq_pos_max_error::KvCacheSeqPosMaxError;
    use std::ptr;

    #[test]
    fn kv_cache_seq_add_status_to_result_maps_every_contract_status() {
        let outcome_0 = kv_cache_seq_add_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_NULL_MODEL,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_0.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_memory_seq_add",
                    detail: "was given a null model argument",
                }
                .into()
            )
        );
        let outcome_1 = kv_cache_seq_add_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_OUT_OF_MEMORY,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_1.err(),
            Some(KvCacheSeqAddError::VendoredOutOfMemory)
        );
    }

    #[test]
    fn kv_cache_seq_div_status_to_result_maps_every_contract_status() {
        let outcome_0 = kv_cache_seq_div_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_NULL_MODEL,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_0.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_memory_seq_div",
                    detail: "was given a null model argument",
                }
                .into()
            )
        );
        let outcome_1 = kv_cache_seq_div_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_OUT_OF_MEMORY,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_1.err(),
            Some(KvCacheSeqDivError::VendoredOutOfMemory)
        );
    }

    #[test]
    fn kv_cache_seq_pos_max_status_to_result_maps_every_contract_status() {
        let outcome_0 = kv_cache_seq_pos_max_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MEMORY_SEQ_POS_MAX_VENDORED_OUT_OF_MEMORY,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_0.err(),
            Some(KvCacheSeqPosMaxError::VendoredOutOfMemory)
        );
    }
}
