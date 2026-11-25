use crate::decision_prompt::DecisionPrompt;

/// The prompts that answer a decision request, and how many leading tokens all of them share,
/// which llama.cpp decodes once and continues from for each prompt.
pub struct DecisionPrompts {
    pub prompts: Vec<DecisionPrompt>,
    pub shared_prefix_length: usize,
}
