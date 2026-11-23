#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatPromptPart {
    Template(String),
    Input(String),
}

impl ChatPromptPart {
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Template(text) | Self::Input(text) => text,
        }
    }
}
