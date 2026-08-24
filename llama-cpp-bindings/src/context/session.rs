use std::ffi::CString;
use std::path::Path;

use crate::context::LlamaContext;
use crate::context::llama_state_seq_flags::LlamaStateSeqFlags;
use crate::context::load_seq_state_error::LoadSeqStateError;
use crate::context::load_session_error::LoadSessionError;
use crate::context::save_seq_state_error::SaveSeqStateError;
use crate::context::save_session_error::SaveSessionError;
use crate::context::state_data_error::StateDataError;
use crate::token::LlamaToken;

fn state_data_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_state_data_status,
    byte_count: usize,
    out_error: *mut std::ffi::c_char,
    operation: &'static str,
) -> Result<usize, StateDataError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_OK => Ok(byte_count),
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_NULL_CTX_ARG => Err(crate::FfiContractError {
            operation,
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_NULL_BUFFER_ARG => {
            Err(crate::FfiContractError {
                operation,
                detail: "was given a null buffer argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_NULL_OUT_BYTE_COUNT_ARG => {
            Err(crate::FfiContractError {
                operation,
                detail: "was given a null out_byte_count argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation,
                detail: "was given a null out_error argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED => {
            Err(StateDataError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_VENDORED_OUT_OF_MEMORY => {
            Err(StateDataError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_STATE_DATA_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                llama_cpp_ffi_status::read_and_free_cpp_string(
                    out_error,
                    operation,
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(StateDataError::Reported { message })
        }
        other => Err(crate::FfiStatusError {
            operation,
            code: i64::from(other),
        }
        .into()),
    }
}

fn process_session_load_result(
    success: bool,
    n_out: usize,
    max_tokens: usize,
    mut tokens: Vec<LlamaToken>,
) -> Result<Vec<LlamaToken>, LoadSessionError> {
    if !success {
        return Err(LoadSessionError::FailedToLoad);
    }

    if n_out > max_tokens {
        return Err(LoadSessionError::InsufficientMaxLength { n_out, max_tokens });
    }

    unsafe { tokens.set_len(n_out) };

    Ok(tokens)
}

fn process_seq_load_result(
    bytes_read: usize,
    n_out: usize,
    max_tokens: usize,
    mut tokens: Vec<LlamaToken>,
) -> Result<(Vec<LlamaToken>, usize), LoadSeqStateError> {
    if bytes_read == 0 {
        return Err(LoadSeqStateError::FailedToLoad);
    }

    if n_out > max_tokens {
        return Err(LoadSeqStateError::InsufficientMaxLength { n_out, max_tokens });
    }

    unsafe { tokens.set_len(n_out) };

    Ok((tokens, bytes_read))
}

impl LlamaContext<'_> {
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to save the state file.
    pub fn state_save_file(
        &self,
        path_session: impl AsRef<Path>,
        tokens: &[LlamaToken],
    ) -> Result<(), SaveSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| SaveSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        if unsafe {
            llama_cpp_bindings_sys::llama_state_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens
                    .as_ptr()
                    .cast::<llama_cpp_bindings_sys::llama_token>(),
                tokens.len(),
            )
        } {
            Ok(())
        } else {
            Err(SaveSessionError::FailedToSave)
        }
    }

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to load the state file.
    pub fn state_load_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, LoadSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| LoadSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        // SAFETY: cast is valid as LlamaToken is repr(transparent)
        let tokens_out = tokens
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token>();

        let success = unsafe {
            llama_cpp_bindings_sys::llama_state_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };
        process_session_load_result(success, n_out, max_tokens, tokens)
    }

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to save the sequence state file.
    ///
    pub fn state_seq_save_file(
        &self,
        filepath: impl AsRef<Path>,
        seq_id: i32,
        tokens: &[LlamaToken],
    ) -> Result<usize, SaveSeqStateError> {
        let path = filepath.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| SaveSeqStateError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        let bytes_written = unsafe {
            llama_cpp_bindings_sys::llama_state_seq_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                seq_id,
                tokens
                    .as_ptr()
                    .cast::<llama_cpp_bindings_sys::llama_token>(),
                tokens.len(),
            )
        };

        if bytes_written == 0 {
            Err(SaveSeqStateError::FailedToSave)
        } else {
            Ok(bytes_written)
        }
    }

    /// # Errors
    ///
    /// Fails if the path is not a valid utf8 or llama.cpp fails to load the sequence state file.
    ///
    pub fn state_seq_load_file(
        &mut self,
        filepath: impl AsRef<Path>,
        dest_seq_id: i32,
        max_tokens: usize,
    ) -> Result<(Vec<LlamaToken>, usize), LoadSeqStateError> {
        let path = filepath.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| LoadSeqStateError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        // SAFETY: cast is valid as LlamaToken is repr(transparent)
        let tokens_out = tokens
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token>();

        let bytes_read = unsafe {
            llama_cpp_bindings_sys::llama_state_seq_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                dest_seq_id,
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };

        process_seq_load_result(bytes_read, n_out, max_tokens, tokens)
    }

    #[must_use]
    pub fn get_state_size(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::llama_state_get_size(self.context.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`StateDataError`] when the vendored serializer fails; the exception is
    /// caught in the C++ wrapper so it can never unwind across the FFI boundary.
    ///
    /// # Safety
    ///
    /// The `dest` buffer must be large enough to hold the complete state data.
    pub unsafe fn copy_state_data(&self, dest: &mut [u8]) -> Result<usize, StateDataError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut std::ffi::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_state_get_data(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        state_data_status_to_result(status, byte_count, out_error, "llama_rs_state_get_data")
    }

    /// # Safety
    ///
    /// The `src` buffer must contain data previously obtained from [`copy_state_data`](Self::copy_state_data)
    /// on a compatible context (same model and parameters). Passing arbitrary or corrupted bytes
    /// will lead to undefined behavior.
    ///
    /// # Errors
    ///
    /// Returns [`StateDataError`] when the vendored deserializer rejects the buffer.
    pub unsafe fn set_state_data(&mut self, src: &[u8]) -> Result<usize, StateDataError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut std::ffi::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_state_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        state_data_status_to_result(status, byte_count, out_error, "llama_rs_state_set_data")
    }

    #[must_use]
    pub fn state_seq_get_size_ext(&self, seq_id: i32, flags: &LlamaStateSeqFlags) -> usize {
        unsafe {
            llama_cpp_bindings_sys::llama_state_seq_get_size_ext(
                self.context.as_ptr(),
                seq_id,
                flags.bits(),
            )
        }
    }

    /// # Safety
    ///
    /// The `dest` buffer must be large enough to hold the complete state data.
    ///
    /// # Errors
    ///
    /// Returns [`StateDataError`] when the vendored serializer fails.
    pub unsafe fn state_seq_get_data_ext(
        &self,
        dest: &mut [u8],
        seq_id: i32,
        flags: &LlamaStateSeqFlags,
    ) -> Result<usize, StateDataError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut std::ffi::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_state_seq_get_data(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
                seq_id,
                flags.bits(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        state_data_status_to_result(status, byte_count, out_error, "llama_rs_state_seq_get_data")
    }

    /// # Safety
    ///
    /// The `src` buffer must contain data previously obtained from
    /// [`state_seq_get_data_ext`](Self::state_seq_get_data_ext) on a compatible context.
    ///
    /// # Errors
    ///
    /// Returns [`StateDataError`] when the vendored deserializer rejects the buffer.
    pub unsafe fn state_seq_set_data_ext(
        &mut self,
        src: &[u8],
        dest_seq_id: i32,
        flags: &LlamaStateSeqFlags,
    ) -> Result<usize, StateDataError> {
        let mut byte_count = 0usize;
        let mut out_error: *mut std::ffi::c_char = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_state_seq_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                dest_seq_id,
                flags.bits(),
                &raw mut byte_count,
                &raw mut out_error,
            )
        };

        state_data_status_to_result(status, byte_count, out_error, "llama_rs_state_seq_set_data")
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::token::LlamaToken;

    use crate::context::load_seq_state_error::LoadSeqStateError;
    use crate::context::load_session_error::LoadSessionError;

    use super::{process_seq_load_result, process_session_load_result};

    #[test]
    fn session_load_success_within_bounds() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(true, 10, 100, tokens);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn session_load_fails_when_not_successful() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(false, 0, 100, tokens);

        assert_eq!(result, Err(LoadSessionError::FailedToLoad));
    }

    #[test]
    fn session_load_fails_when_n_out_exceeds_max() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_session_load_result(true, 101, 100, tokens);

        assert_eq!(
            result,
            Err(LoadSessionError::InsufficientMaxLength {
                n_out: 101,
                max_tokens: 100,
            })
        );
    }

    #[test]
    fn seq_load_success_within_bounds() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(42, 10, 100, tokens);

        assert!(result.is_ok());
        let (loaded, bytes) = result.unwrap();
        assert_eq!(loaded.len(), 10);
        assert_eq!(bytes, 42);
    }

    #[test]
    fn seq_load_fails_when_zero_bytes_read() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(0, 0, 100, tokens);

        assert_eq!(result, Err(LoadSeqStateError::FailedToLoad));
    }

    #[test]
    fn seq_load_fails_when_n_out_exceeds_max() {
        let tokens = vec![LlamaToken::new(0); 100];
        let result = process_seq_load_result(42, 101, 100, tokens);

        assert_eq!(
            result,
            Err(LoadSeqStateError::InsufficientMaxLength {
                n_out: 101,
                max_tokens: 100,
            })
        );
    }
}
