use llama_cpp_model::chat_templates::ChatTemplates;

#[derive(Clone, Copy, Debug)]
pub enum ConversationTemplates<'templates> {
    Absent,
    Present(&'templates ChatTemplates),
}
