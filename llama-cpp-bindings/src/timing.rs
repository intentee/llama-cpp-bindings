//! Safe wrapper around `llama_timings`.
use std::fmt::{Debug, Display, Formatter};

/// A wrapper around `llama_timings`.
#[derive(Clone, Copy, Debug)]
pub struct LlamaTimings {
    /// The underlying `llama_perf_context_data` from the C API.
    pub timings: llama_cpp_bindings_sys::llama_perf_context_data,
}

impl LlamaTimings {
    /// Create a new `LlamaTimings`.
    /// ```
    /// # use llama_cpp_bindings::timing::LlamaTimings;
    /// let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5, 6, 1);
    /// let timings_str = "load time = 2.00 ms
    /// prompt eval time = 3.00 ms / 5 tokens (0.60 ms per token, 1666.67 tokens per second)
    /// eval time = 4.00 ms / 6 runs (0.67 ms per token, 1500.00 tokens per second)\n";
    /// assert_eq!(timings_str, format!("{}", timings));
    /// ```
    #[must_use]
    pub fn new(
        t_start_ms: f64,
        t_load_ms: f64,
        t_p_eval_ms: f64,
        t_eval_ms: f64,
        n_p_eval: i32,
        n_eval: i32,
        n_reused: i32,
    ) -> Self {
        Self {
            timings: llama_cpp_bindings_sys::llama_perf_context_data {
                t_start_ms,
                t_load_ms,
                t_p_eval_ms,
                t_eval_ms,
                n_p_eval,
                n_eval,
                n_reused,
            },
        }
    }

    /// Get the start time in milliseconds.
    #[must_use]
    pub fn t_start_ms(&self) -> f64 {
        self.timings.t_start_ms
    }

    /// Get the load time in milliseconds.
    #[must_use]
    pub fn t_load_ms(&self) -> f64 {
        self.timings.t_load_ms
    }

    /// Get the prompt evaluation time in milliseconds.
    #[must_use]
    pub fn t_p_eval_ms(&self) -> f64 {
        self.timings.t_p_eval_ms
    }

    /// Get the evaluation time in milliseconds.
    #[must_use]
    pub fn t_eval_ms(&self) -> f64 {
        self.timings.t_eval_ms
    }

    /// Get the number of prompt evaluations.
    #[must_use]
    pub fn n_p_eval(&self) -> i32 {
        self.timings.n_p_eval
    }

    /// Get the number of evaluations.
    #[must_use]
    pub fn n_eval(&self) -> i32 {
        self.timings.n_eval
    }

    /// Set the start time in milliseconds.
    pub fn set_t_start_ms(&mut self, t_start_ms: f64) {
        self.timings.t_start_ms = t_start_ms;
    }

    /// Set the load time in milliseconds.
    pub fn set_t_load_ms(&mut self, t_load_ms: f64) {
        self.timings.t_load_ms = t_load_ms;
    }

    /// Set the prompt evaluation time in milliseconds.
    pub fn set_t_p_eval_ms(&mut self, t_p_eval_ms: f64) {
        self.timings.t_p_eval_ms = t_p_eval_ms;
    }

    /// Set the evaluation time in milliseconds.
    pub fn set_t_eval_ms(&mut self, t_eval_ms: f64) {
        self.timings.t_eval_ms = t_eval_ms;
    }

    /// Set the number of prompt evaluations.
    pub fn set_n_p_eval(&mut self, n_p_eval: i32) {
        self.timings.n_p_eval = n_p_eval;
    }

    /// Set the number of evaluations.
    pub fn set_n_eval(&mut self, n_eval: i32) {
        self.timings.n_eval = n_eval;
    }
}

impl Display for LlamaTimings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "load time = {:.2} ms", self.t_load_ms())?;

        if self.n_p_eval() > 0 {
            writeln!(
                f,
                "prompt eval time = {:.2} ms / {} tokens ({:.2} ms per token, {:.2} tokens per second)",
                self.t_p_eval_ms(),
                self.n_p_eval(),
                self.t_p_eval_ms() / f64::from(self.n_p_eval()),
                1e3 / self.t_p_eval_ms() * f64::from(self.n_p_eval())
            )?;
        } else {
            writeln!(
                f,
                "prompt eval time = {:.2} ms / 0 tokens",
                self.t_p_eval_ms(),
            )?;
        }

        if self.n_eval() > 0 {
            writeln!(
                f,
                "eval time = {:.2} ms / {} runs ({:.2} ms per token, {:.2} tokens per second)",
                self.t_eval_ms(),
                self.n_eval(),
                self.t_eval_ms() / f64::from(self.n_eval()),
                1e3 / self.t_eval_ms() * f64::from(self.n_eval())
            )?;
        } else {
            writeln!(f, "eval time = {:.2} ms / 0 runs", self.t_eval_ms(),)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaTimings;

    #[test]
    fn new_and_getters_roundtrip() {
        let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5, 6, 1);

        assert_eq!(timings.t_start_ms(), 1.0);
        assert_eq!(timings.t_load_ms(), 2.0);
        assert_eq!(timings.t_p_eval_ms(), 3.0);
        assert_eq!(timings.t_eval_ms(), 4.0);
        assert_eq!(timings.n_p_eval(), 5);
        assert_eq!(timings.n_eval(), 6);
    }

    #[test]
    fn setters_modify_values() {
        let mut timings = LlamaTimings::new(0.0, 0.0, 0.0, 0.0, 0, 0, 0);

        timings.set_t_start_ms(10.0);
        timings.set_t_load_ms(20.0);
        timings.set_t_p_eval_ms(30.0);
        timings.set_t_eval_ms(40.0);
        timings.set_n_p_eval(50);
        timings.set_n_eval(60);

        assert_eq!(timings.t_start_ms(), 10.0);
        assert_eq!(timings.t_load_ms(), 20.0);
        assert_eq!(timings.t_p_eval_ms(), 30.0);
        assert_eq!(timings.t_eval_ms(), 40.0);
        assert_eq!(timings.n_p_eval(), 50);
        assert_eq!(timings.n_eval(), 60);
    }

    #[test]
    fn display_format_with_valid_counts() {
        let timings = LlamaTimings::new(1.0, 2.0, 3.0, 4.0, 5, 6, 1);
        let output = format!("{timings}");

        assert!(output.contains("load time = 2.00 ms"));
        assert!(output.contains("prompt eval time = 3.00 ms / 5 tokens"));
        assert!(output.contains("eval time = 4.00 ms / 6 runs"));
    }

    #[test]
    fn display_format_handles_zero_eval_counts() {
        let timings = LlamaTimings::new(0.0, 1.0, 2.0, 3.0, 0, 0, 0);
        let output = format!("{timings}");

        assert!(output.contains("load time = 1.00 ms"));
        assert!(output.contains("prompt eval time = 2.00 ms / 0 tokens"));
        assert!(output.contains("eval time = 3.00 ms / 0 runs"));
        assert!(!output.contains("NaN"));
        assert!(!output.contains("inf"));
    }
}
