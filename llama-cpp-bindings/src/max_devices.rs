#[must_use]
pub fn max_devices() -> usize {
    unsafe { llama_cpp_bindings_sys::llama_max_devices() }
}

#[cfg(test)]
mod tests {
    use crate::max_devices::max_devices;

    #[test]
    fn returns_at_least_one() {
        let device_count = max_devices();

        assert!(device_count >= 1);
    }
}
