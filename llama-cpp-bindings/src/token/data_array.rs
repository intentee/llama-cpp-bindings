use std::ptr;

use crate::error::SamplerApplyError;
use crate::error::TokenSamplingError;
use crate::sampling::LlamaSampler;
use crate::token::data::LlamaTokenData;

use super::LlamaToken;
use llama_cpp_ffi_status::read_and_free_cpp_string;

fn sampler_apply_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_sampler_apply_status,
    out_error: *mut std::os::raw::c_char,
) -> Result<(), SamplerApplyError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_NULL_SAMPLER_ARG => {
            Err(SamplerApplyError::NullSampler)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(SamplerApplyError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_sampler_apply",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(SamplerApplyError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_NULL_DATA_ARRAY_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_apply",
                detail: "was given a null data_array argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_apply",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_sampler_apply",
            code: i64::from(other),
        }
        .into()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenDataArray {
    pub data: Vec<LlamaTokenData>,
    pub selected: Option<usize>,
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    #[must_use]
    pub const fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    pub fn from_iter<TIterator>(data: TIterator, sorted: bool) -> Self
    where
        TIterator: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }
}

impl LlamaTokenDataArray {
    /// # Errors
    ///
    /// Returns [`crate::FfiContractError`] when the vendored sampler grows the array beyond the
    /// capacity this buffer was allocated with, which would make the following `set_len`
    /// undefined behaviour.
    ///
    /// # Safety
    ///
    /// The returned array formed by the data pointer and the length must entirely consist of
    /// initialized token data.
    /// If the data is not sorted, sorted must be false.
    pub unsafe fn modify_as_c_llama_token_data_array<TResult>(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_bindings_sys::llama_token_data_array) -> TResult,
    ) -> Result<TResult, crate::FfiContractError> {
        let size = self.data.len();
        let data = self
            .data
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token_data>();

        let mut c_llama_token_data_array = llama_cpp_bindings_sys::llama_token_data_array {
            data,
            size,
            selected: self
                .selected
                .and_then(|selected_index| selected_index.try_into().ok())
                .unwrap_or(-1),
            sorted: self.sorted,
        };

        let result = modify(&mut c_llama_token_data_array);

        if c_llama_token_data_array.size > self.data.capacity() {
            return Err(crate::FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the vendored sampler grew the token data array beyond its capacity",
            });
        }

        unsafe {
            if !ptr::eq(c_llama_token_data_array.data, data) {
                ptr::copy(
                    c_llama_token_data_array.data,
                    data,
                    c_llama_token_data_array.size,
                );
            }
            self.data.set_len(c_llama_token_data_array.size);
        }

        self.sorted = c_llama_token_data_array.sorted;
        self.selected = c_llama_token_data_array
            .selected
            .try_into()
            .ok()
            .filter(|&selected_index| selected_index < self.data.len());

        Ok(result)
    }

    /// # Errors
    ///
    /// Returns [`SamplerApplyError`] if the sampler pointer is null, the vendored
    /// sampler runs out of memory, or it throws a C++ exception while applying.
    pub fn apply_sampler(&mut self, sampler: &LlamaSampler) -> Result<(), SamplerApplyError> {
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                let mut out_error: *mut std::os::raw::c_char = ptr::null_mut();
                let status = llama_cpp_bindings_sys::llama_rs_sampler_apply(
                    sampler.as_ptr(),
                    c_llama_token_data_array,
                    &raw mut out_error,
                );
                sampler_apply_status_to_result(status, out_error)
            })?
        }
    }

    /// # Errors
    /// Returns [`SamplerApplyError`] if applying the sampler fails.
    pub fn with_sampler(mut self, sampler: &mut LlamaSampler) -> Result<Self, SamplerApplyError> {
        self.apply_sampler(sampler)?;
        Ok(self)
    }

    /// # Errors
    /// Returns [`TokenSamplingError::SamplerApply`] if applying the sampler fails, or
    /// [`TokenSamplingError::NoTokenSelected`] if the sampler fails to select a token.
    pub fn sample_token(&mut self, seed: u32) -> Result<LlamaToken, TokenSamplingError> {
        let sampler = LlamaSampler::dist(seed)?;
        self.apply_sampler(&sampler)?;
        self.selected_token()
            .ok_or(TokenSamplingError::NoTokenSelected)
    }

    /// # Errors
    /// Returns [`TokenSamplingError::SamplerApply`] if applying the sampler fails, or
    /// [`TokenSamplingError::NoTokenSelected`] if the sampler fails to select a token.
    pub fn sample_token_greedy(&mut self) -> Result<LlamaToken, TokenSamplingError> {
        let sampler = LlamaSampler::greedy()?;
        self.apply_sampler(&sampler)?;
        self.selected_token()
            .ok_or(TokenSamplingError::NoTokenSelected)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::SamplerApplyError;
    use crate::token::LlamaToken;
    use crate::token::data::LlamaTokenData;

    use super::LlamaTokenDataArray;
    use super::sampler_apply_status_to_result;

    #[test]
    fn sampler_apply_status_allocation_failed_returns_not_enough_memory() {
        assert_eq!(
            sampler_apply_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_ERROR_STRING_ALLOCATION_FAILED,
                std::ptr::null_mut(),
            ),
            Err(SamplerApplyError::NotEnoughMemory),
        );
    }

    #[test]
    fn sampler_apply_status_cxx_exception_without_a_message_is_a_contract_error() {
        assert_eq!(
            sampler_apply_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_SAMPLER_APPLY_VENDORED_THREW_CXX_EXCEPTION,
                std::ptr::null_mut(),
            ),
            Err(crate::FfiContractError {
                operation: "llama_rs_sampler_apply",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()),
        );
    }

    #[test]
    fn sampler_apply_unknown_status_is_preserved() {
        let result = sampler_apply_status_to_result(255, std::ptr::null_mut());

        assert_eq!(
            result,
            Err(SamplerApplyError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_sampler_apply",
                code: 255,
            }))
        );
    }

    #[test]
    fn apply_greedy_sampler_selects_highest_logit() {
        use crate::sampling::LlamaSampler;

        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
            ],
            false,
        );

        let sampler = LlamaSampler::greedy().expect("greedy sampler must initialize");
        array
            .apply_sampler(&sampler)
            .expect("test: greedy sampler must apply");

        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn with_sampler_builder_pattern() {
        use crate::sampling::LlamaSampler;

        let mut sampler = LlamaSampler::greedy().expect("greedy sampler must initialize");
        let array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 5.0, 0.0),
            ],
            false,
        )
        .with_sampler(&mut sampler)
        .expect("test: building with greedy sampler must succeed");

        assert_eq!(array.selected_token(), Some(LlamaToken::new(1)));
    }

    #[test]
    fn sample_token_greedy_returns_highest() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(10), 0.1, 0.0),
                LlamaTokenData::new(LlamaToken::new(20), 9.9, 0.0),
            ],
            false,
        );

        let token = array
            .sample_token_greedy()
            .expect("test: greedy sampler should select a token");

        assert_eq!(token, LlamaToken::new(20));
    }

    #[test]
    fn from_iter_creates_array_from_iterator() {
        let array = LlamaTokenDataArray::from_iter(
            [
                LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(2), 2.0, 0.0),
            ],
            false,
        );

        assert_eq!(array.data.len(), 3);
        assert!(!array.sorted);
        assert!(array.selected.is_none());
    }

    #[test]
    fn sample_token_with_seed_selects_a_token() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(10), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(20), 1.0, 0.0),
            ],
            false,
        );

        let token = array
            .sample_token(42)
            .expect("test: dist sampler should select a token");

        assert!(token == LlamaToken::new(10) || token == LlamaToken::new(20));
    }

    #[test]
    fn selected_token_returns_none_when_no_selection() {
        let array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        assert!(array.selected_token().is_none());
    }

    #[test]
    fn selected_token_returns_none_when_index_out_of_bounds() {
        let array = LlamaTokenDataArray {
            data: vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            selected: Some(5),
            sorted: false,
        };

        assert!(array.selected_token().is_none());
    }

    #[test]
    fn modify_as_c_llama_token_data_array_copies_when_data_pointer_changes() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 2.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(2), 3.0, 0.0),
            ],
            false,
        );

        let replacement = [
            llama_cpp_bindings_sys::llama_token_data {
                id: 10,
                logit: 5.0,
                p: 0.0,
            },
            llama_cpp_bindings_sys::llama_token_data {
                id: 20,
                logit: 6.0,
                p: 0.0,
            },
        ];

        unsafe {
            array
                .modify_as_c_llama_token_data_array(|c_array| {
                    c_array.data = replacement.as_ptr().cast_mut();
                    c_array.size = replacement.len();
                    c_array.selected = 0;
                })
                .expect("the replacement fits within the allocated capacity");
        }

        assert_eq!(array.data.len(), 2);
        assert_eq!(array.data[0].id(), LlamaToken::new(10));
        assert_eq!(array.data[1].id(), LlamaToken::new(20));
        assert_eq!(array.selected, Some(0));
    }

    #[test]
    fn modify_clears_selection_when_index_is_out_of_range() {
        let mut array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        unsafe {
            array
                .modify_as_c_llama_token_data_array(|c_array| {
                    c_array.selected = 5;
                })
                .expect("the array is left at its original size");
        }

        assert_eq!(array.selected, None);
    }

    #[test]
    fn selected_overflow_uses_negative_one() {
        let mut array = LlamaTokenDataArray {
            data: vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            selected: Some(usize::MAX),
            sorted: false,
        };

        unsafe {
            array
                .modify_as_c_llama_token_data_array(|c_array| {
                    assert_eq!(c_array.selected, -1);
                })
                .expect("the array is left at its original size");
        }
    }

    #[test]
    fn oversized_result_is_reported_as_a_contract_error() {
        let mut array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );
        let capacity = array.data.capacity();

        let result = unsafe {
            array.modify_as_c_llama_token_data_array(|c_array| {
                c_array.size = capacity + 1;
            })
        };

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the vendored sampler grew the token data array beyond its capacity",
            })
        );
        assert_eq!(array.data.len(), 1);
    }

    #[test]
    fn preset_valid_selection_is_passed_through_as_index() {
        let mut array = LlamaTokenDataArray {
            data: vec![
                LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(1), 2.0, 0.0),
            ],
            selected: Some(1),
            sorted: false,
        };

        unsafe {
            array
                .modify_as_c_llama_token_data_array(|c_array| {
                    assert_eq!(c_array.selected, 1);
                })
                .expect("the array is left at its original size");
        }

        assert_eq!(array.selected, Some(1));
    }
}
