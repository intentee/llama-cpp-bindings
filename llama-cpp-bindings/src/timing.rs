use std::fmt::{Display, Formatter, Write};

use llama_cpp_bindings_sys::llama_perf_context_data;

fn write_timings(timings: &LlamaTimings, writer: &mut dyn Write) -> std::fmt::Result {
    writeln!(writer, "load time = {:.2} ms", timings.t_load_ms())?;

    if timings.n_p_eval() > 0 {
        writeln!(
            writer,
            "prompt eval time = {:.2} ms / {} tokens ({:.2} ms per token, {:.2} tokens per second)",
            timings.t_p_eval_ms(),
            timings.n_p_eval(),
            timings.t_p_eval_ms() / f64::from(timings.n_p_eval()),
            1e3 / timings.t_p_eval_ms() * f64::from(timings.n_p_eval())
        )?;
    } else {
        writeln!(
            writer,
            "prompt eval time = {:.2} ms / 0 tokens",
            timings.t_p_eval_ms(),
        )?;
    }

    if timings.n_eval() > 0 {
        writeln!(
            writer,
            "eval time = {:.2} ms / {} runs ({:.2} ms per token, {:.2} tokens per second)",
            timings.t_eval_ms(),
            timings.n_eval(),
            timings.t_eval_ms() / f64::from(timings.n_eval()),
            1e3 / timings.t_eval_ms() * f64::from(timings.n_eval())
        )?;
    } else {
        writeln!(writer, "eval time = {:.2} ms / 0 runs", timings.t_eval_ms())?;
    }

    writeln!(writer, "graphs reused = {}", timings.n_reused())?;
    writeln!(
        writer,
        "graph reallocs = {} with the same size / {} with a changed size",
        timings.n_reallocs_same(),
        timings.n_reallocs_diff()
    )?;

    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct LlamaTimings {
    timings: llama_perf_context_data,
}

impl From<llama_perf_context_data> for LlamaTimings {
    fn from(timings: llama_perf_context_data) -> Self {
        Self { timings }
    }
}

impl LlamaTimings {
    #[must_use]
    pub const fn t_start_ms(&self) -> f64 {
        self.timings.t_start_ms
    }

    #[must_use]
    pub const fn t_load_ms(&self) -> f64 {
        self.timings.t_load_ms
    }

    #[must_use]
    pub const fn t_p_eval_ms(&self) -> f64 {
        self.timings.t_p_eval_ms
    }

    #[must_use]
    pub const fn t_eval_ms(&self) -> f64 {
        self.timings.t_eval_ms
    }

    #[must_use]
    pub const fn n_p_eval(&self) -> i32 {
        self.timings.n_p_eval
    }

    #[must_use]
    pub const fn n_eval(&self) -> i32 {
        self.timings.n_eval
    }

    #[must_use]
    pub const fn n_reused(&self) -> i32 {
        self.timings.n_reused
    }

    #[must_use]
    pub const fn n_reallocs_same(&self) -> i32 {
        self.timings.n_reallocs_same
    }

    #[must_use]
    pub const fn n_reallocs_diff(&self) -> i32 {
        self.timings.n_reallocs_diff
    }
}

impl Display for LlamaTimings {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write_timings(self, formatter)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use llama_cpp_bindings_sys::llama_perf_context_data;

    use super::LlamaTimings;
    use super::write_timings;

    struct WriterFailingAfterWrites {
        remaining_writes: usize,
    }

    impl Write for WriterFailingAfterWrites {
        fn write_str(&mut self, _text: &str) -> std::fmt::Result {
            if self.remaining_writes == 0 {
                return Err(std::fmt::Error);
            }
            self.remaining_writes -= 1;

            Ok(())
        }
    }

    struct WriterCountingWrites {
        writes: usize,
    }

    impl Write for WriterCountingWrites {
        fn write_str(&mut self, _text: &str) -> std::fmt::Result {
            self.writes += 1;

            Ok(())
        }
    }

    const fn timings_with_eval_counts(n_p_eval: i32, n_eval: i32) -> LlamaTimings {
        LlamaTimings {
            timings: llama_perf_context_data {
                t_start_ms: 1.0,
                t_load_ms: 2.0,
                t_p_eval_ms: 3.0,
                t_eval_ms: 4.0,
                n_p_eval,
                n_eval,
                n_reused: 7,
                n_reallocs_same: 8,
                n_reallocs_diff: 9,
            },
        }
    }

    #[test]
    fn display_reports_throughput_of_evaluated_prompt_and_generated_tokens() {
        let output = timings_with_eval_counts(5, 6).to_string();

        assert_eq!(
            output,
            "load time = 2.00 ms\n\
             prompt eval time = 3.00 ms / 5 tokens (0.60 ms per token, 1666.67 tokens per second)\n\
             eval time = 4.00 ms / 6 runs (0.67 ms per token, 1500.00 tokens per second)\n\
             graphs reused = 7\n\
             graph reallocs = 8 with the same size / 9 with a changed size\n"
        );
    }

    #[test]
    fn display_omits_throughput_when_nothing_was_evaluated() {
        let output = timings_with_eval_counts(0, 0).to_string();

        assert_eq!(
            output,
            "load time = 2.00 ms\n\
             prompt eval time = 3.00 ms / 0 tokens\n\
             eval time = 4.00 ms / 0 runs\n\
             graphs reused = 7\n\
             graph reallocs = 8 with the same size / 9 with a changed size\n"
        );
    }

    #[test]
    fn write_timings_propagates_a_writer_failure_at_every_write() {
        for timings in [
            timings_with_eval_counts(5, 6),
            timings_with_eval_counts(0, 0),
        ] {
            let mut counting_writer = WriterCountingWrites { writes: 0 };
            assert_eq!(write_timings(&timings, &mut counting_writer), Ok(()));

            for remaining_writes in 0..counting_writer.writes {
                assert_eq!(
                    write_timings(&timings, &mut WriterFailingAfterWrites { remaining_writes }),
                    Err(std::fmt::Error)
                );
            }
        }
    }
}
