use llama_cpp_model::llama_token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatGrammarTrigger {
    Token(LlamaToken),
    Word(String),
    Pattern(String),
    PatternFull(String),
}
