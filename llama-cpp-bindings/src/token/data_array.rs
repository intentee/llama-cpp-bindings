use std::ptr;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;

use super::LlamaToken;
use crate::token::data::LlamaTokenData;

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

    pub fn replace_candidates<TIterator>(&mut self, candidates: TIterator)
    where
        TIterator: IntoIterator<Item = LlamaTokenData>,
    {
        self.data.clear();
        self.data.extend(candidates);
        self.selected = None;
        self.sorted = false;
    }

    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }

    /// Whether a sampler left at least one candidate it could still select.
    #[must_use]
    pub fn has_selectable_candidate(&self) -> bool {
        self.data
            .iter()
            .any(|candidate| candidate.logit() > f32::NEG_INFINITY)
    }
}

impl LlamaTokenDataArray {
    fn c_selection(&self) -> Result<i64, FfiContractError> {
        let Some(selected_index) = self.selected else {
            return Ok(-1);
        };

        self.data
            .get(selected_index)
            .and_then(|_selected_candidate| i64::try_from(selected_index).ok())
            .ok_or(FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the token data array selects a token outside itself",
            })
    }

    fn selection_of(&self, c_selected: i64) -> Result<Option<usize>, FfiContractError> {
        if c_selected == -1 {
            return Ok(None);
        }

        usize::try_from(c_selected)
            .ok()
            .filter(|&selected_index| selected_index < self.data.len())
            .map(Some)
            .ok_or(FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the llama.cpp sampler selected a token outside the token data array",
            })
    }

    /// # Errors
    ///
    /// Returns [`llama_cpp_ffi_status::ffi_contract_error::FfiContractError`] when the llama.cpp sampler grows the array beyond the
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
    ) -> Result<TResult, FfiContractError> {
        let size = self.data.len();
        let data = self
            .data
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token_data>();

        let mut c_llama_token_data_array = llama_cpp_bindings_sys::llama_token_data_array {
            data,
            size,
            selected: self.c_selection()?,
            sorted: self.sorted,
        };

        let result = modify(&mut c_llama_token_data_array);

        if c_llama_token_data_array.size > self.data.capacity() {
            return Err(FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the llama.cpp sampler grew the token data array beyond its capacity",
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
        self.selected = self.selection_of(c_llama_token_data_array.selected)?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;

    use super::LlamaTokenDataArray;
    use crate::token::LlamaToken;
    use crate::token::data::LlamaTokenData;

    #[test]
    fn selected_token_returns_none_when_no_selection() {
        let array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        assert_eq!(array.selected_token(), None);
    }

    #[test]
    fn selected_token_returns_none_when_index_out_of_bounds() {
        let array = LlamaTokenDataArray {
            data: vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            selected: Some(5),
            sorted: false,
        };

        assert_eq!(array.selected_token(), None);
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
    fn a_sampler_selecting_a_token_outside_the_array_is_a_contract_violation() {
        let mut array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            false,
        );

        let result = unsafe {
            array.modify_as_c_llama_token_data_array(|c_array| {
                c_array.selected = 5;
            })
        };

        assert_eq!(
            result,
            Err(FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the llama.cpp sampler selected a token outside the token data array",
            })
        );
    }

    #[test]
    fn a_selection_outside_the_array_is_refused_before_reaching_llama_cpp() {
        for selected in [1, usize::MAX] {
            let mut array = LlamaTokenDataArray {
                data: vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
                selected: Some(selected),
                sorted: false,
            };

            let result = unsafe { array.modify_as_c_llama_token_data_array(|_c_array| ()) };

            assert_eq!(
                result,
                Err(FfiContractError {
                    operation: "modify_as_c_llama_token_data_array",
                    detail: "the token data array selects a token outside itself",
                })
            );
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
            Err(FfiContractError {
                operation: "modify_as_c_llama_token_data_array",
                detail: "the llama.cpp sampler grew the token data array beyond its capacity",
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
