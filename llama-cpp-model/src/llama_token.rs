use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LlamaToken(pub llama_cpp_bindings_sys::llama_token);

impl Display for LlamaToken {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl LlamaToken {
    #[must_use]
    pub const fn new(token_id: i32) -> Self {
        Self(token_id)
    }

    /// Whether the token is one of the `n_vocab` tokens of a vocabulary.
    #[must_use]
    pub const fn is_within_vocabulary(self, n_vocab: i32) -> bool {
        0 <= self.0 && self.0 < n_vocab
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
    fn only_the_ids_from_zero_up_to_the_vocabulary_size_are_within_the_vocabulary() {
        assert_eq!(
            [-1, 0, 99, 100].map(|token_id| LlamaToken::new(token_id).is_within_vocabulary(100)),
            [false, true, true, false]
        );
    }
}
