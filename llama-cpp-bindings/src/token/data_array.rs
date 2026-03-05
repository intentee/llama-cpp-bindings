//! an rusty equivalent of `llama_token_data_array`.
use std::ptr;

use crate::{sampling::LlamaSampler, token::data::LlamaTokenData};

use super::LlamaToken;

/// a safe wrapper around `llama_token_data_array`.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenDataArray {
    /// the underlying data
    pub data: Vec<LlamaTokenData>,
    /// the index of the selected token in ``data``
    pub selected: Option<usize>,
    /// is the data sorted?
    pub sorted: bool,
}

impl LlamaTokenDataArray {
    /// Create a new `LlamaTokenDataArray` from a vector and whether or not the data is sorted.
    ///
    /// ```
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// # use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// let array = LlamaTokenDataArray::new(vec![
    ///         LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///         LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    ///    ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    /// ```
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>, sorted: bool) -> Self {
        Self {
            data,
            selected: None,
            sorted,
        }
    }

    /// Create a new `LlamaTokenDataArray` from an iterator and whether or not the data is sorted.
    /// ```
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// # use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// let array = LlamaTokenDataArray::from_iter([
    ///     LlamaTokenData::new(LlamaToken(0), 0.0, 0.0),
    ///     LlamaTokenData::new(LlamaToken(1), 0.1, 0.1)
    /// ], false);
    /// assert_eq!(array.data.len(), 2);
    /// assert_eq!(array.sorted, false);
    pub fn from_iter<TIterator>(data: TIterator, sorted: bool) -> LlamaTokenDataArray
    where
        TIterator: IntoIterator<Item = LlamaTokenData>,
    {
        Self::new(data.into_iter().collect(), sorted)
    }

    /// Returns the current selected token, if one exists.
    #[must_use]
    pub fn selected_token(&self) -> Option<LlamaToken> {
        self.data.get(self.selected?).map(LlamaTokenData::id)
    }
}

impl LlamaTokenDataArray {
    /// Modify the underlying data as a `llama_token_data_array`. and reconstruct the `LlamaTokenDataArray`.
    ///
    /// # Panics
    ///
    /// Panics if some of the safety conditions are not met. (we cannot check all of them at
    /// runtime so breaking them is UB)
    ///
    /// # Safety
    ///
    /// The returned array formed by the data pointer and the length must entirely consist of
    /// initialized token data and the length must be less than the capacity of this array's data
    /// buffer.
    /// If the data is not sorted, sorted must be false.
    pub unsafe fn modify_as_c_llama_token_data_array<TResult>(
        &mut self,
        modify: impl FnOnce(&mut llama_cpp_bindings_sys::llama_token_data_array) -> TResult,
    ) -> TResult {
        let size = self.data.len();
        let data = self
            .data
            .as_mut_ptr()
            .cast::<llama_cpp_bindings_sys::llama_token_data>();

        let mut c_llama_token_data_array = llama_cpp_bindings_sys::llama_token_data_array {
            data,
            size,
            selected: self.selected.and_then(|s| s.try_into().ok()).unwrap_or(-1),
            sorted: self.sorted,
        };

        let result = modify(&mut c_llama_token_data_array);

        assert!(
            c_llama_token_data_array.size <= self.data.capacity(),
            "Size of the returned array exceeds the data buffer's capacity!"
        );
        // SAFETY: caller guarantees the returned data and size are valid.
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
            .filter(|&s| s < self.data.len());

        result
    }

    /// Modifies the data array by applying a sampler to it
    pub fn apply_sampler(&mut self, sampler: &LlamaSampler) {
        unsafe {
            self.modify_as_c_llama_token_data_array(|c_llama_token_data_array| {
                llama_cpp_bindings_sys::llama_sampler_apply(
                    sampler.sampler,
                    c_llama_token_data_array,
                );
            });
        }
    }

    /// Modifies the data array by applying a sampler to it
    #[must_use]
    pub fn with_sampler(mut self, sampler: &mut LlamaSampler) -> Self {
        self.apply_sampler(sampler);
        self
    }

    /// Randomly selects a token from the candidates based on their probabilities.
    ///
    /// # Panics
    /// If the internal llama.cpp sampler fails to select a token.
    pub fn sample_token(&mut self, seed: u32) -> LlamaToken {
        self.apply_sampler(&LlamaSampler::dist(seed));
        self.selected_token()
            .expect("Dist sampler failed to select a token!")
    }

    /// Selects the token with the highest probability.
    ///
    /// # Panics
    /// If the internal llama.cpp sampler fails to select a token.
    pub fn sample_token_greedy(&mut self) -> LlamaToken {
        self.apply_sampler(&LlamaSampler::greedy());
        self.selected_token()
            .expect("Greedy sampler failed to select a token!")
    }
}

#[cfg(test)]
mod tests {
    use crate::token::LlamaToken;
    use crate::token::data::LlamaTokenData;

    use super::LlamaTokenDataArray;

    #[test]
    fn new_empty() {
        let array = LlamaTokenDataArray::new(vec![], false);

        assert!(array.data.is_empty());
        assert!(!array.sorted);
        assert_eq!(array.selected, None);
    }

    #[test]
    fn new_with_sorted_flag() {
        let array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 1.0, 0.0)],
            true,
        );

        assert!(array.sorted);
    }

    #[test]
    fn from_iter_collects_all_elements() {
        let items = [
            LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(1), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken::new(2), 2.0, 0.0),
        ];
        let array = LlamaTokenDataArray::from_iter(items, false);

        assert_eq!(array.data.len(), 3);
        assert_eq!(array.data[0].id(), LlamaToken::new(0));
        assert_eq!(array.data[2].id(), LlamaToken::new(2));
    }

    #[test]
    fn selected_token_none_when_no_selection() {
        let array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(5), 1.0, 0.0)],
            false,
        );

        assert_eq!(array.selected_token(), None);
    }

    #[test]
    fn selected_token_returns_correct_token_when_selected() {
        let mut array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken::new(10), 0.0, 0.0),
                LlamaTokenData::new(LlamaToken::new(20), 1.0, 0.0),
            ],
            false,
        );
        array.selected = Some(1);

        assert_eq!(array.selected_token(), Some(LlamaToken::new(20)));
    }

    #[test]
    fn selected_token_none_when_index_out_of_bounds() {
        let mut array = LlamaTokenDataArray::new(
            vec![LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0)],
            false,
        );
        array.selected = Some(5);

        assert_eq!(array.selected_token(), None);
    }
}
