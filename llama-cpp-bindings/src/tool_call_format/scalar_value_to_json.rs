#[must_use]
pub fn scalar_value_to_json(raw: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(raw)
        .unwrap_or_else(|_not_json| serde_json::Value::String(raw.to_owned()))
}
