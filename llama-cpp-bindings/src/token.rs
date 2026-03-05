//! Safe wrappers around `llama_token_data` and `llama_token_data_array`.

use std::fmt::Debug;
use std::fmt::Display;

pub mod data;
pub mod data_array;
pub mod logit_bias;

/// A safe wrapper for `llama_token`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LlamaToken(pub llama_cpp_bindings_sys::llama_token);

impl Display for LlamaToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LlamaToken {
    /// Create a new `LlamaToken` from a i32.
    ///
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// let token = LlamaToken::new(0);
    /// assert_eq!(token, LlamaToken(0));
    /// ```
    #[must_use]
    pub fn new(token_id: i32) -> Self {
        Self(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaToken;

    #[test]
    fn display_shows_inner_value() {
        let token = LlamaToken::new(42);

        assert_eq!(format!("{token}"), "42");
    }

    #[test]
    fn display_negative_value() {
        let token = LlamaToken::new(-1);

        assert_eq!(format!("{token}"), "-1");
    }
}
