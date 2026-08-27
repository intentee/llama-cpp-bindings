#[derive(Debug, Eq, PartialEq)]
pub struct SeparatorSplit<'body> {
    pub before: &'body str,
    pub after: &'body str,
}

impl<'body> SeparatorSplit<'body> {
    #[must_use]
    pub fn at_first(input: &'body str, separator: &str) -> Option<Self> {
        input
            .split_once(separator)
            .map(|(before, after)| Self { before, after })
    }
}
