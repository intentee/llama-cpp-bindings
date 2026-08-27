#[must_use]
pub fn consume_optional_prefix<'body>(input: &'body str, literal: &str) -> &'body str {
    input.strip_prefix(literal).unwrap_or(input)
}
