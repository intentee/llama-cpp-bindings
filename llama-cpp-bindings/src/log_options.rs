/// Options to configure how llama.cpp logs are intercepted.
#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    pub disabled: bool,
}

impl LogOptions {
    /// If enabled, logs are sent to tracing. If disabled, all logs are suppressed. Default is for
    /// logs to be sent to tracing.
    #[must_use]
    pub fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;
        self
    }
}
