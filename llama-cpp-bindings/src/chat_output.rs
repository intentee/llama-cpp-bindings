use std::ops::Range;

use llama_cpp_bindings_sys::LLAMA_TOKEN_NULL;
use llama_cpp_bindings_sys::llama_token;

use crate::token::LlamaToken;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChatOutput {
    text: String,
    byte_tokens: Vec<llama_token>,
}

impl From<&str> for ChatOutput {
    /// Holds `text` without tokens, the way `common_chat_input` holds plain text.
    fn from(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            byte_tokens: vec![LLAMA_TOKEN_NULL; text.len()],
        }
    }
}

impl ChatOutput {
    /// Appends the text `token` decoded to, recording the token at the first byte of its piece
    /// the way `common_chat_input` lays tokens over the text, and returns the bytes the piece
    /// occupies.
    pub fn push(&mut self, token: LlamaToken, piece: &str) -> Range<usize> {
        let start = self.text.len();

        self.text.push_str(piece);
        self.byte_tokens.extend((0..piece.len()).map(|offset| {
            if offset == 0 {
                token.0
            } else {
                LLAMA_TOKEN_NULL
            }
        }));

        start..self.text.len()
    }

    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub fn byte_tokens(&self) -> &[llama_token] {
        &self.byte_tokens
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_sys::LLAMA_TOKEN_NULL;

    use crate::chat_output::ChatOutput;
    use crate::token::LlamaToken;

    #[test]
    fn each_token_owns_the_bytes_of_its_piece_and_marks_its_first_byte() {
        let mut output = ChatOutput::default();

        let pieces = [
            output.push(LlamaToken::new(7), "<a>"),
            output.push(LlamaToken::new(9), ""),
            output.push(LlamaToken::new(11), "hi"),
        ];

        assert_eq!(pieces, [0..3, 3..3, 3..5]);
        assert_eq!(output.text(), "<a>hi");
        assert_eq!(
            output.byte_tokens(),
            [7, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL, 11, LLAMA_TOKEN_NULL]
        );
    }

    #[test]
    fn plain_text_marks_no_byte_with_a_token() {
        let output = ChatOutput::from("ab");

        assert_eq!(output.byte_tokens(), [LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL]);
    }
}
