/// The beginning of an assistant answer the model continues: its reasoning, closed, and the
/// content written so far.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatAssistantPrefill {
    pub reasoning_content: String,
    pub content: String,
}
