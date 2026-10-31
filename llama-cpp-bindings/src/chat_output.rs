use llama_cpp_bindings_sys::LLAMA_TOKEN_NULL;
use llama_cpp_bindings_sys::llama_token;

use crate::chat_output_token::ChatOutputToken;
use crate::token::LlamaToken;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChatOutput {
    text: String,
    byte_tokens: Vec<llama_token>,
    tokens: Vec<ChatOutputToken>,
}

impl From<&str> for ChatOutput {
    /// Holds `text` without tokens, the way `common_chat_input` holds plain text.
    fn from(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            byte_tokens: vec![LLAMA_TOKEN_NULL; text.len()],
            tokens: Vec::new(),
        }
    }
}

impl ChatOutput {
    /// Appends the text `token` decoded to, recording the token at the first byte of its piece
    /// the way `common_chat_input` lays tokens over the text.
    pub fn push(&mut self, token: LlamaToken, piece: &str) {
        let start = self.text.len();

        self.text.push_str(piece);
        self.byte_tokens.extend((0..piece.len()).map(|offset| {
            if offset == 0 {
                token.0
            } else {
                LLAMA_TOKEN_NULL
            }
        }));
        self.tokens.push(ChatOutputToken {
            token,
            bytes: start..self.text.len(),
        });
    }

    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub fn byte_tokens(&self) -> &[llama_token] {
        &self.byte_tokens
    }

    #[must_use]
    pub fn tokens(&self) -> &[ChatOutputToken] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_sys::LLAMA_TOKEN_NULL;

    use crate::chat_output::ChatOutput;
    use crate::chat_output_token::ChatOutputToken;
    use crate::token::LlamaToken;

    #[test]
    fn each_token_owns_the_bytes_of_its_piece_and_marks_its_first_byte() {
        let mut output = ChatOutput::default();

        output.push(LlamaToken::new(7), "<a>");
        output.push(LlamaToken::new(9), "");
        output.push(LlamaToken::new(11), "hi");

        assert_eq!(output.text(), "<a>hi");
        assert_eq!(
            output.byte_tokens(),
            [7, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL, 11, LLAMA_TOKEN_NULL]
        );
        assert_eq!(
            output.tokens(),
            [
                ChatOutputToken {
                    token: LlamaToken::new(7),
                    bytes: 0..3,
                },
                ChatOutputToken {
                    token: LlamaToken::new(9),
                    bytes: 3..3,
                },
                ChatOutputToken {
                    token: LlamaToken::new(11),
                    bytes: 3..5,
                },
            ]
        );
    }

    #[test]
    fn plain_text_carries_no_tokens() {
        let output = ChatOutput::from("ab");

        assert_eq!(output.byte_tokens(), [LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL]);
        assert_eq!(output.tokens(), []);
    }
}
