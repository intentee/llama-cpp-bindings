//! Safe wrapper around `llama_token_data`.
use crate::token::LlamaToken;

/// A transparent wrapper around `llama_token_data`.
///
/// Do not rely on `repr(transparent)` for this type. It should be considered an implementation
/// detail and may change across minor versions.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LlamaTokenData {
    data: llama_cpp_bindings_sys::llama_token_data,
}

impl LlamaTokenData {
    /// Create a new token data from a token, logit, and probability.
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let token_data = LlamaTokenData::new(token, 1.0, 1.0);
    #[must_use]
    pub fn new(LlamaToken(id): LlamaToken, logit: f32, p: f32) -> Self {
        LlamaTokenData {
            data: llama_cpp_bindings_sys::llama_token_data { id, logit, p },
        }
    }
    /// Get the token's id
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let token_data = LlamaTokenData::new(token, 1.0, 1.0);
    /// assert_eq!(token_data.id(), token);
    /// ```
    #[must_use]
    pub fn id(&self) -> LlamaToken {
        LlamaToken(self.data.id)
    }

    /// Get the token's logit
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let token_data = LlamaTokenData::new(token, 1.0, 1.0);
    /// assert_eq!(token_data.logit(), 1.0);
    /// ```
    #[must_use]
    pub fn logit(&self) -> f32 {
        self.data.logit
    }

    /// Get the token's probability
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let token_data = LlamaTokenData::new(token, 1.0, 1.0);
    /// assert_eq!(token_data.p(), 1.0);
    /// ```
    #[must_use]
    pub fn p(&self) -> f32 {
        self.data.p
    }

    /// Set the token's id
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let mut token_data = LlamaTokenData::new(token, 1.0, 1.0);
    /// token_data.set_id(LlamaToken::new(2));
    /// assert_eq!(token_data.id(), LlamaToken::new(2));
    /// ```
    pub fn set_id(&mut self, id: LlamaToken) {
        self.data.id = id.0;
    }

    /// Set the token's logit
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let mut token_data = LlamaTokenData::new(token, 1.0, 1.0);
    /// token_data.set_logit(2.0);
    /// assert_eq!(token_data.logit(), 2.0);
    /// ```
    pub fn set_logit(&mut self, logit: f32) {
        self.data.logit = logit;
    }

    /// Set the token's probability
    /// ```
    /// # use llama_cpp_bindings::token::LlamaToken;
    /// # use llama_cpp_bindings::token::data::LlamaTokenData;
    /// let token = LlamaToken::new(1);
    /// let mut token_data = LlamaTokenData::new(token, 1.0, 1.0);
    /// token_data.set_p(2.0);
    /// assert_eq!(token_data.p(), 2.0);
    /// ```
    pub fn set_p(&mut self, p: f32) {
        self.data.p = p;
    }
}

#[cfg(test)]
mod tests {
    use crate::token::LlamaToken;

    use super::LlamaTokenData;

    #[test]
    fn new_and_getters_roundtrip() {
        let token = LlamaToken::new(7);
        let data = LlamaTokenData::new(token, 0.5, 0.25);

        assert_eq!(data.id(), token);
        assert_eq!(data.logit(), 0.5);
        assert_eq!(data.p(), 0.25);
    }

    #[test]
    fn setters_modify_values() {
        let mut data = LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0);

        data.set_id(LlamaToken::new(42));
        data.set_logit(3.14);
        data.set_p(0.99);

        assert_eq!(data.id(), LlamaToken::new(42));
        assert_eq!(data.logit(), 3.14);
        assert_eq!(data.p(), 0.99);
    }

    #[test]
    fn boundary_values() {
        let data = LlamaTokenData::new(LlamaToken::new(i32::MAX), f32::MAX, f32::MIN);

        assert_eq!(data.id(), LlamaToken::new(i32::MAX));
        assert_eq!(data.logit(), f32::MAX);
        assert_eq!(data.p(), f32::MIN);
    }

    #[test]
    fn zero_values() {
        let data = LlamaTokenData::new(LlamaToken::new(0), 0.0, 0.0);

        assert_eq!(data.id(), LlamaToken::new(0));
        assert_eq!(data.logit(), 0.0);
        assert_eq!(data.p(), 0.0);
    }
}
