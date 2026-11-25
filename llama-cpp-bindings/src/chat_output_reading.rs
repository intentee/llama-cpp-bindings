use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_output::ChatOutput;
use llama_cpp_chat::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_chat::chat_reading::ChatReading;
use llama_cpp_chat::chat_session::ChatSession;

pub struct ChatOutputReading {
    pub reading: ChatReading,
    pub output_length: usize,
    pub completeness: ChatOutputCompleteness,
}

impl ChatOutputReading {
    pub fn read(
        session: &ChatSession,
        output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> Result<Self, ChatError> {
        Ok(Self {
            reading: session.read(output, completeness)?,
            output_length: output.text().len(),
            completeness,
        })
    }

    pub fn has_read(
        &self,
        growing_output: &ChatOutput,
        completeness: ChatOutputCompleteness,
    ) -> bool {
        self.output_length == growing_output.text().len() && self.completeness == completeness
    }
}
