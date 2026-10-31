#[derive(Debug, Eq, PartialEq)]
pub struct ParsedPrefix<'body, TValue> {
    pub value: TValue,
    pub remaining: &'body str,
}
