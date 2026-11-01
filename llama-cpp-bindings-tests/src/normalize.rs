#[must_use]
pub fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |accumulator, &value| value.mul_add(value, accumulator))
        .sqrt();

    input.iter().map(|&value| value / magnitude).collect()
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn scales_a_vector_to_unit_length() {
        assert_eq!(normalize(&[3.0, 4.0]), vec![0.6, 0.8]);
    }
}
