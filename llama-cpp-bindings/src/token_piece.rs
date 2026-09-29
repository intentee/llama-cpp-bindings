#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenPiece {
    Marker(String),
    Visible(String),
}

impl TokenPiece {
    #[must_use]
    pub fn raw(&self) -> &str {
        match self {
            Self::Marker(text) | Self::Visible(text) => text,
        }
    }

    #[must_use]
    pub fn visible(&self) -> &str {
        match self {
            Self::Marker(_) => "",
            Self::Visible(text) => text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TokenPiece;

    #[test]
    fn a_marker_is_raw_text_without_a_visible_part() {
        let piece = TokenPiece::Marker("<think>".to_owned());

        assert_eq!(piece.raw(), "<think>");
        assert_eq!(piece.visible(), "");
    }

    #[test]
    fn visible_text_is_both_raw_and_visible() {
        let piece = TokenPiece::Visible("hello".to_owned());

        assert_eq!(piece.raw(), "hello");
        assert_eq!(piece.visible(), "hello");
    }
}
