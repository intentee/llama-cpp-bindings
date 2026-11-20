use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_copy_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunk_view::MtmdInputChunkView;

/// # Safety
///
/// On `LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK` `out_chunk` must be the non-null chunk llama.cpp copied,
/// and on `LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the
/// non-null message the C++ side allocated; the message is freed here.
unsafe fn input_chunk_copy_status_to_result(
    status: llama_rs_mtmd_input_chunk_copy_status,
    out_chunk: *mut llama_cpp_bindings_sys::bindings::mtmd_input_chunk,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::mtmd_input_chunk>, MtmdError> {
    match status {
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_chunk) })
        }
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::ChunkCopyNotEnoughMemory)
        }
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::ChunkCopyLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_input_chunk_copy", |message| {
                    MtmdError::ChunkCopyReported { message }
                }))
        }
    }
}

/// A chunk this binding owns and frees when dropped, so it outlives the chunk list it was copied
/// from, while it stays tied to the projector that tokenized it.
#[derive(Debug)]
pub struct MtmdInputChunk<'context> {
    chunk: NonNull<llama_cpp_bindings_sys::bindings::mtmd_input_chunk>,
    mtmd_context: &'context MtmdContext,
}

impl<'context> MtmdInputChunk<'context> {
    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to copy the chunk.
    pub fn copy(chunk: MtmdInputChunkView<'_, 'context>) -> Result<Self, MtmdError> {
        let mut out_chunk: *mut llama_cpp_bindings_sys::bindings::mtmd_input_chunk =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_copy(
                chunk.as_ptr(),
                &raw mut out_chunk,
                &raw mut out_error,
            )
        };

        unsafe { input_chunk_copy_status_to_result(status, out_chunk, out_error) }.map(|copied| {
            Self {
                chunk: copied,
                mtmd_context: chunk.mtmd_context(),
            }
        })
    }

    #[must_use]
    pub const fn view(&self) -> MtmdInputChunkView<'_, 'context> {
        unsafe { MtmdInputChunkView::from_raw(self.chunk, self.mtmd_context) }
    }
}

impl Drop for MtmdInputChunk<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_input_chunk_free(self.chunk.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_copy_status;

    use super::input_chunk_copy_status_to_result;
    use crate::mtmd_error::MtmdError;

    #[test]
    fn chunk_copy_memory_failures_map_to_their_errors() {
        let statuses = [
            llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED,
            llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY,
        ];

        assert_eq!(
            statuses.map(|status| {
                unsafe {
                    input_chunk_copy_status_to_result(status, ptr::null_mut(), ptr::null_mut())
                }
                .err()
            }),
            [
                Some(MtmdError::ChunkCopyNotEnoughMemory),
                Some(MtmdError::ChunkCopyLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn a_thrown_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"allocation threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                input_chunk_copy_status_to_result(
                    llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                )
            }
            .err(),
            Some(MtmdError::ChunkCopyReported {
                message: "allocation threw".to_owned(),
            })
        );
    }
}
