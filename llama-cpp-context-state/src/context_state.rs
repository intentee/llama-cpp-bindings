use std::ffi::c_char;
use std::path::Path;
use std::ptr;

use enumflags2::BitFlags;

use llama_cpp_bindings_sys::bindings::llama_rs_state_data_status;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_path_c_string::path_c_string::path_c_string;

use crate::context_state_error::ContextStateError;
use crate::llama_state_seq_flag::LlamaStateSeqFlag;
use crate::loaded_seq_state::LoadedSeqState;
use crate::read_exception_message::read_exception_message;

/// # Safety
///
/// On `LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message
/// the C++ side allocated; it is freed here.
unsafe fn state_data_status_to_result(
    status: llama_rs_state_data_status,
    byte_count: usize,
    out_error: *mut c_char,
    operation: &'static str,
) -> Result<usize, ContextStateError> {
    match status {
        llama_rs_state_data_status::LLAMA_RS_STATE_DATA_OK => Ok(byte_count),
        llama_rs_state_data_status::LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ContextStateError::StateDataNotEnoughMemory)
        }
        llama_rs_state_data_status::LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ContextStateError::StateDataLlamaCppOutOfMemory)
        }
        llama_rs_state_data_status::LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_exception_message(operation, out_error) }?;

            Err(ContextStateError::StateDataReported { message })
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message
/// the C++ side allocated; it is freed here.
unsafe fn state_data_restore_status_to_result(
    status: llama_rs_state_data_status,
    restored_bytes: usize,
    provided_bytes: usize,
    out_error: *mut c_char,
    operation: &'static str,
) -> Result<usize, ContextStateError> {
    let restored =
        unsafe { state_data_status_to_result(status, restored_bytes, out_error, operation) }?;

    if restored == 0 && provided_bytes != 0 {
        return Err(ContextStateError::StateDataNothingRestored { provided_bytes });
    }

    Ok(restored)
}

fn process_session_load_result(
    success: bool,
    n_out: usize,
    mut tokens: Vec<LlamaToken>,
) -> Result<Vec<LlamaToken>, ContextStateError> {
    if !success {
        return Err(ContextStateError::SessionFileNotLoaded);
    }

    tokens.truncate(n_out);

    Ok(tokens)
}

fn process_seq_load_result(
    bytes_read: usize,
    n_out: usize,
    mut tokens: Vec<LlamaToken>,
) -> Result<LoadedSeqState, ContextStateError> {
    if bytes_read == 0 {
        return Err(ContextStateError::SeqStateFileNotLoaded);
    }

    tokens.truncate(n_out);

    Ok(LoadedSeqState { tokens, bytes_read })
}

pub trait ContextState {
    /// # Errors
    ///
    /// Returns [`ContextStateError`] when the llama.cpp serializer fails; the exception is
    /// caught on the C++ side so it can never unwind across the FFI boundary.
    ///
    /// # Safety
    ///
    /// The `dest` buffer must be large enough to hold the complete state data.
    unsafe fn state_get_data(&self, dest: &mut [u8]) -> Result<usize, ContextStateError>;

    #[must_use]
    fn state_get_size(&self) -> usize;

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to load the state file.
    fn state_load_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, ContextStateError>;

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to save the state file.
    fn state_save_file(
        &self,
        path_session: impl AsRef<Path>,
        tokens: &[LlamaToken],
    ) -> Result<(), ContextStateError>;

    /// # Safety
    ///
    /// The `dest` buffer must be large enough to hold the complete state data.
    ///
    /// # Errors
    ///
    /// Returns [`ContextStateError`] when the llama.cpp serializer fails.
    unsafe fn state_seq_get_data_ext(
        &self,
        dest: &mut [u8],
        seq_id: i32,
        flags: BitFlags<LlamaStateSeqFlag>,
    ) -> Result<usize, ContextStateError>;

    #[must_use]
    fn state_seq_get_size_ext(&self, seq_id: i32, flags: BitFlags<LlamaStateSeqFlag>) -> usize;

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to load the sequence state file.
    ///
    fn state_seq_load_file(
        &mut self,
        filepath: impl AsRef<Path>,
        dest_seq_id: i32,
        max_tokens: usize,
    ) -> Result<LoadedSeqState, ContextStateError>;

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to save the sequence state file.
    ///
    fn state_seq_save_file(
        &self,
        filepath: impl AsRef<Path>,
        seq_id: i32,
        tokens: &[LlamaToken],
    ) -> Result<usize, ContextStateError>;

    /// # Safety
    ///
    /// The `src` buffer must contain data previously obtained from
    /// [`state_seq_get_data_ext`](Self::state_seq_get_data_ext) on a compatible context.
    ///
    /// # Errors
    ///
    /// Returns [`ContextStateError`] when the llama.cpp deserializer rejects the buffer.
    unsafe fn state_seq_set_data_ext(
        &mut self,
        src: &[u8],
        dest_seq_id: i32,
        flags: BitFlags<LlamaStateSeqFlag>,
    ) -> Result<usize, ContextStateError>;

    /// # Safety
    ///
    /// The `src` buffer must contain data previously obtained from [`state_get_data`](Self::state_get_data)
    /// on a compatible context (same model and parameters). Passing arbitrary or corrupted bytes
    /// will lead to undefined behavior.
    ///
    /// # Errors
    ///
    /// Returns [`ContextStateError`] when the llama.cpp deserializer rejects the buffer.
    unsafe fn state_set_data(&mut self, src: &[u8]) -> Result<usize, ContextStateError>;
}

impl ContextState for LlamaContext<'_> {
    unsafe fn state_get_data(&self, dest: &mut [u8]) -> Result<usize, ContextStateError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_state_get_data(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        unsafe {
            state_data_status_to_result(status, byte_count, out_error, "llama_rs_state_get_data")
        }
    }

    fn state_get_size(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::bindings::llama_state_get_size(self.context.as_ptr()) }
    }

    fn state_load_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, ContextStateError> {
        let cstr = path_c_string(path_session.as_ref())
            .map_err(ContextStateError::LoadSessionPathInvalid)?;
        let mut tokens = vec![LlamaToken::new(0); max_tokens];
        let mut n_out = 0;

        let tokens_out = tokens
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::bindings::llama_token>();

        let success = unsafe {
            llama_cpp_bindings_sys::bindings::llama_state_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };
        process_session_load_result(success, n_out, tokens)
    }

    fn state_save_file(
        &self,
        path_session: impl AsRef<Path>,
        tokens: &[LlamaToken],
    ) -> Result<(), ContextStateError> {
        let cstr = path_c_string(path_session.as_ref())
            .map_err(ContextStateError::SaveSessionPathInvalid)?;

        if unsafe {
            llama_cpp_bindings_sys::bindings::llama_state_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens
                    .as_ptr()
                    .cast::<llama_cpp_bindings_sys::bindings::llama_token>(),
                tokens.len(),
            )
        } {
            Ok(())
        } else {
            Err(ContextStateError::SessionFileNotSaved)
        }
    }

    unsafe fn state_seq_get_data_ext(
        &self,
        dest: &mut [u8],
        seq_id: i32,
        flags: BitFlags<LlamaStateSeqFlag>,
    ) -> Result<usize, ContextStateError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_state_seq_get_data(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
                seq_id,
                flags.bits(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        unsafe {
            state_data_status_to_result(
                status,
                byte_count,
                out_error,
                "llama_rs_state_seq_get_data",
            )
        }
    }

    fn state_seq_get_size_ext(&self, seq_id: i32, flags: BitFlags<LlamaStateSeqFlag>) -> usize {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_state_seq_get_size_ext(
                self.context.as_ptr(),
                seq_id,
                flags.bits(),
            )
        }
    }

    fn state_seq_load_file(
        &mut self,
        filepath: impl AsRef<Path>,
        dest_seq_id: i32,
        max_tokens: usize,
    ) -> Result<LoadedSeqState, ContextStateError> {
        let cstr =
            path_c_string(filepath.as_ref()).map_err(ContextStateError::LoadSeqStatePathInvalid)?;
        let mut tokens = vec![LlamaToken::new(0); max_tokens];
        let mut n_out = 0;

        let tokens_out = tokens
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::bindings::llama_token>();

        let bytes_read = unsafe {
            llama_cpp_bindings_sys::bindings::llama_state_seq_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                dest_seq_id,
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };

        process_seq_load_result(bytes_read, n_out, tokens)
    }

    fn state_seq_save_file(
        &self,
        filepath: impl AsRef<Path>,
        seq_id: i32,
        tokens: &[LlamaToken],
    ) -> Result<usize, ContextStateError> {
        let cstr =
            path_c_string(filepath.as_ref()).map_err(ContextStateError::SaveSeqStatePathInvalid)?;

        let bytes_written = unsafe {
            llama_cpp_bindings_sys::bindings::llama_state_seq_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                seq_id,
                tokens
                    .as_ptr()
                    .cast::<llama_cpp_bindings_sys::bindings::llama_token>(),
                tokens.len(),
            )
        };

        if bytes_written == 0 {
            Err(ContextStateError::SeqStateFileNotSaved)
        } else {
            Ok(bytes_written)
        }
    }

    unsafe fn state_seq_set_data_ext(
        &mut self,
        src: &[u8],
        dest_seq_id: i32,
        flags: BitFlags<LlamaStateSeqFlag>,
    ) -> Result<usize, ContextStateError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_state_seq_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                dest_seq_id,
                flags.bits(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        unsafe {
            state_data_restore_status_to_result(
                status,
                byte_count,
                src.len(),
                out_error,
                "llama_rs_state_seq_set_data",
            )
        }
    }

    unsafe fn state_set_data(&mut self, src: &[u8]) -> Result<usize, ContextStateError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_state_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        unsafe {
            state_data_restore_status_to_result(
                status,
                byte_count,
                src.len(),
                out_error,
                "llama_rs_state_set_data",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_state_data_status;
    use llama_cpp_model::llama_token::LlamaToken;

    use super::process_seq_load_result;
    use super::process_session_load_result;
    use super::state_data_restore_status_to_result;
    use super::state_data_status_to_result;
    use crate::context_state_error::ContextStateError;
    use crate::loaded_seq_state::LoadedSeqState;

    #[test]
    fn session_load_truncates_the_tokens_to_the_loaded_count() {
        let tokens: Vec<LlamaToken> = (0..100).map(LlamaToken::new).collect();

        assert_eq!(
            process_session_load_result(true, 10, tokens),
            Ok((0..10).map(LlamaToken::new).collect())
        );
    }

    #[test]
    fn session_load_fails_when_not_successful() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(false, 0, tokens);

        assert_eq!(result, Err(ContextStateError::SessionFileNotLoaded));
    }

    #[test]
    fn seq_load_truncates_the_tokens_to_the_loaded_count_and_reports_the_bytes_read() {
        let tokens: Vec<LlamaToken> = (0..100).map(LlamaToken::new).collect();

        assert_eq!(
            process_seq_load_result(42, 10, tokens),
            Ok(LoadedSeqState {
                tokens: (0..10).map(LlamaToken::new).collect(),
                bytes_read: 42,
            })
        );
    }

    #[test]
    fn seq_load_fails_when_zero_bytes_read() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(0, 0, tokens);

        assert_eq!(result, Err(ContextStateError::SeqStateFileNotLoaded));
    }

    #[test]
    fn state_data_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            state_data_status_to_result(
                llama_rs_state_data_status::LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
                "",
            )
        };
        assert_eq!(
            outcome.err(),
            Some(ContextStateError::StateDataLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn state_data_status_to_result_reports_not_enough_memory_for_the_error_string() {
        assert_eq!(
            unsafe {
                state_data_status_to_result(
                    llama_rs_state_data_status::LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED,
                    0,
                    ptr::null_mut(),
                    "llama_rs_state_get_data",
                )
            },
            Err(ContextStateError::StateDataNotEnoughMemory)
        );
    }

    #[test]
    fn state_data_status_to_result_reports_an_exception_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"serializer threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                state_data_status_to_result(
                    llama_rs_state_data_status::LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    0,
                    out_error,
                    "llama_rs_state_get_data",
                )
            },
            Err(ContextStateError::StateDataReported {
                message: "serializer threw".to_owned()
            })
        );
    }

    #[test]
    fn restoring_state_data_fails_only_when_a_non_empty_snapshot_restores_nothing() {
        assert_eq!(
            [(0, 16), (0, 0), (16, 16)].map(|(restored_bytes, provided_bytes)| {
                unsafe {
                    state_data_restore_status_to_result(
                        llama_rs_state_data_status::LLAMA_RS_STATE_DATA_OK,
                        restored_bytes,
                        provided_bytes,
                        ptr::null_mut(),
                        "llama_rs_state_seq_set_data",
                    )
                }
            }),
            [
                Err(ContextStateError::StateDataNothingRestored { provided_bytes: 16 }),
                Ok(0),
                Ok(16),
            ]
        );
    }

    #[test]
    fn restoring_state_data_propagates_a_failed_status() {
        assert_eq!(
            unsafe {
                state_data_restore_status_to_result(
                    llama_rs_state_data_status::LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY,
                    0,
                    16,
                    ptr::null_mut(),
                    "llama_rs_state_seq_set_data",
                )
            },
            Err(ContextStateError::StateDataLlamaCppOutOfMemory)
        );
    }
}
