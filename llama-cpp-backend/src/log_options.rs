use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_void;
use std::sync::PoisonError;
use std::sync::{Mutex, OnceLock};

use log::log;
use log::warn;

use llama_cpp_log_decoder::decode_anomaly::DecodeAnomaly;
use llama_cpp_log_decoder::decode_output::DecodeOutput;
use llama_cpp_log_decoder::incoming_log_level::IncomingLogLevel;
use llama_cpp_log_decoder::log_decoder::LogDecoder;
use llama_cpp_log_decoder::log_level::LogLevel;
use llama_cpp_log_decoder::log_line::LogLine;

use crate::log_record::LogRecord;

static LLAMA_SOURCE: OnceLock<LogSource> = OnceLock::new();
static GGML_SOURCE: OnceLock<LogSource> = OnceLock::new();

const fn ggml_level_to_incoming(
    raw: llama_cpp_bindings_sys::bindings::ggml_log_level,
) -> IncomingLogLevel {
    match raw {
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE => IncomingLogLevel::None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_DEBUG => IncomingLogLevel::Debug,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_INFO => IncomingLogLevel::Info,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_WARN => IncomingLogLevel::Warn,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_ERROR => IncomingLogLevel::Error,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_CONT => IncomingLogLevel::Cont,
        other => IncomingLogLevel::Unknown(other),
    }
}

fn resolve_record(LogLine { level, text }: LogLine, demote_info_to_debug: bool) -> LogRecord {
    let effective_level =
        if demote_info_to_debug && matches!(level, LogLevel::Info | LogLevel::None) {
            LogLevel::Debug
        } else {
            level
        };

    match effective_level {
        LogLevel::Debug => LogRecord {
            level: log::Level::Debug,
            text,
        },
        LogLevel::Info | LogLevel::None => LogRecord {
            level: log::Level::Info,
            text,
        },
        LogLevel::Warn => LogRecord {
            level: log::Level::Warn,
            text,
        },
        LogLevel::Error => LogRecord {
            level: log::Level::Error,
            text,
        },
        LogLevel::Unknown(raw) => LogRecord {
            level: log::Level::Warn,
            text: format!("[unknown level {raw}] {text}"),
        },
    }
}

fn dispatch_line(source: &LogSource, line: LogLine) {
    let LogRecord { level, text } = resolve_record(line, source.options().demote_info_to_debug);
    log!(target: source.target, level, "{text}");
}

fn dispatch_output(source: &LogSource, output: DecodeOutput) {
    match output {
        DecodeOutput::None => {}
        DecodeOutput::Line(line) => dispatch_line(source, line),
        DecodeOutput::TwoLines { earlier, current } => {
            dispatch_line(source, earlier);
            dispatch_line(source, current);
        }
    }
}

fn dispatch_anomaly(LogSource { target, .. }: &LogSource, anomaly: DecodeAnomaly) {
    warn!(
        target: target,
        "llama.cpp log decoder anomaly: {anomaly:?}",
    );
}

unsafe extern "C" fn logs_to_log(
    raw_level: llama_cpp_bindings_sys::bindings::ggml_log_level,
    text_ptr: *const c_char,
    data_ptr: *mut c_void,
) {
    let source: &LogSource = unsafe { &*data_ptr.cast::<LogSource>() };

    if source.options().disabled {
        return;
    }

    if text_ptr.is_null() {
        warn!(
            target: source.target,
            "received NULL text pointer from llama.cpp log callback",
        );
        return;
    }

    let text_cstr = unsafe { CStr::from_ptr(text_ptr) };
    let text = text_cstr.to_string_lossy();

    let incoming = ggml_level_to_incoming(raw_level);

    let result = {
        let mut decoder = source
            .decoder
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        decoder.feed(incoming, &text)
    };

    dispatch_output(source, result.output);

    if let Some(anomaly) = result.anomaly {
        dispatch_anomaly(source, anomaly);
    }
}

struct LogSource {
    decoder: Mutex<LogDecoder>,
    target: &'static str,
    options: Mutex<LogOptions>,
}

impl LogSource {
    const fn new(target: &'static str, options: LogOptions) -> Self {
        Self {
            decoder: Mutex::new(LogDecoder::new()),
            target,
            options: Mutex::new(options),
        }
    }

    fn options(&self) -> LogOptions {
        self.options
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn replace_options(&self, options: LogOptions) {
        *self.options.lock().unwrap_or_else(PoisonError::into_inner) = options;
    }
}

#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    pub disabled: bool,
    pub demote_info_to_debug: bool,
}

impl LogOptions {
    #[must_use]
    pub const fn with_demote_info_to_debug(mut self, demote: bool) -> Self {
        self.demote_info_to_debug = demote;

        self
    }

    #[must_use]
    pub const fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;

        self
    }

    pub fn send_logs_to_log(self) {
        let llama_source =
            LLAMA_SOURCE.get_or_init(|| LogSource::new("llama.cpp", Self::default()));
        let ggml_source = GGML_SOURCE.get_or_init(|| LogSource::new("ggml", Self::default()));

        llama_source.replace_options(self.clone());
        ggml_source.replace_options(self);

        let llama_source: *const LogSource = llama_source;
        let ggml_source: *const LogSource = ggml_source;

        unsafe {
            llama_cpp_bindings_sys::bindings::llama_log_set(
                Some(logs_to_log),
                llama_source.cast::<c_void>().cast_mut(),
            );
            llama_cpp_bindings_sys::bindings::ggml_log_set(
                Some(logs_to_log),
                ggml_source.cast::<c_void>().cast_mut(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::PoisonError;
    use std::thread;

    use log::Level;

    use llama_cpp_capturing_logger::captured_record::CapturedRecord;
    use llama_cpp_capturing_logger::capturing_logger::CapturingLogger;
    use llama_cpp_log_decoder::decode_output::DecodeOutput;
    use llama_cpp_log_decoder::incoming_log_level::IncomingLogLevel;
    use llama_cpp_log_decoder::log_level::LogLevel;
    use llama_cpp_log_decoder::log_line::LogLine;

    use super::LogOptions;
    use super::LogSource;
    use super::dispatch_output;
    use super::ggml_level_to_incoming;
    use super::logs_to_log;
    use super::resolve_record;
    use crate::log_record::LogRecord;

    fn records_for(target: &str) -> Vec<LogRecord> {
        CapturingLogger::records_for(target)
            .into_iter()
            .map(|CapturedRecord { level, message, .. }| LogRecord {
                level,
                text: message,
            })
            .collect()
    }

    fn invoke_callback(
        level: llama_cpp_bindings_sys::bindings::ggml_log_level,
        text: &CStr,
        source: &LogSource,
    ) {
        let ptr = ptr::from_ref(source).cast::<c_void>().cast_mut();
        unsafe {
            logs_to_log(level, text.as_ptr(), ptr);
        }
    }

    #[test]
    fn every_ggml_log_level_maps_to_its_incoming_level() {
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE),
            IncomingLogLevel::None,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_DEBUG),
            IncomingLogLevel::Debug,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_INFO),
            IncomingLogLevel::Info,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_WARN),
            IncomingLogLevel::Warn,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_ERROR),
            IncomingLogLevel::Error,
        );
        assert_eq!(
            ggml_level_to_incoming(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_CONT),
            IncomingLogLevel::Cont,
        );
    }

    #[test]
    fn an_unknown_ggml_log_level_keeps_its_raw_value() {
        assert_eq!(
            ggml_level_to_incoming(9999),
            IncomingLogLevel::Unknown(9999)
        );
    }

    #[test]
    fn disabled_logs_emit_no_record() {
        CapturingLogger::install();

        let target = "test-dispatch-when-disabled";
        let source = LogSource::new(target, LogOptions::default().with_logs_enabled(false));
        invoke_callback(
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_INFO,
            c"hello\n",
            &source,
        );

        assert_eq!(records_for(target), Vec::<LogRecord>::new());
    }

    #[test]
    fn demoting_info_logs_emits_info_as_debug() {
        CapturingLogger::install();

        let target = "test-demote-info-on-info";
        let source = LogSource::new(
            target,
            LogOptions::default().with_demote_info_to_debug(true),
        );
        invoke_callback(
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_INFO,
            c"info-line\n",
            &source,
        );

        assert_eq!(
            records_for(target),
            [LogRecord {
                level: Level::Debug,
                text: "info-line".to_owned(),
            }]
        );
    }

    #[test]
    fn demoting_info_logs_keeps_warnings_as_warnings() {
        CapturingLogger::install();

        let target = "test-demote-info-on-warn";
        let source = LogSource::new(
            target,
            LogOptions::default().with_demote_info_to_debug(true),
        );
        invoke_callback(
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_WARN,
            c"warn-line\n",
            &source,
        );

        assert_eq!(
            records_for(target),
            [LogRecord {
                level: Level::Warn,
                text: "warn-line".to_owned(),
            }]
        );
    }

    #[test]
    fn an_unknown_level_is_logged_as_a_warning_naming_it() {
        CapturingLogger::install();

        let target = "test-dispatch-unknown-level";
        let source = LogSource::new(target, LogOptions::default());
        invoke_callback(9999, c"weird\n", &source);

        assert_eq!(
            records_for(target),
            [LogRecord {
                level: Level::Warn,
                text: "[unknown level 9999] weird".to_owned(),
            }]
        );
    }

    #[test]
    fn an_orphan_continuation_is_logged_with_an_anomaly_warning() {
        CapturingLogger::install();

        let target = "test-dispatch-orphan-cont";
        let source = LogSource::new(target, LogOptions::default());
        invoke_callback(
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_CONT,
            c"ghost\n",
            &source,
        );

        assert_eq!(
            records_for(target),
            [
                LogRecord {
                    level: Level::Info,
                    text: "ghost".to_owned(),
                },
                LogRecord {
                    level: Level::Warn,
                    text: "llama.cpp log decoder anomaly: OrphanCont".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn resolve_record_error_level_maps_to_error_level() {
        assert_eq!(
            resolve_record(
                LogLine {
                    level: LogLevel::Error,
                    text: "boom".to_owned(),
                },
                false,
            ),
            LogRecord {
                level: Level::Error,
                text: "boom".to_owned(),
            }
        );
    }

    #[test]
    fn dispatch_output_none_emits_no_records() {
        CapturingLogger::install();

        let target = "test-dispatch-output-none";
        let source = LogSource::new(target, LogOptions::default());
        dispatch_output(&source, DecodeOutput::None);

        assert_eq!(records_for(target), Vec::<LogRecord>::new());
    }

    #[test]
    fn dispatch_output_two_lines_emits_both_records() {
        CapturingLogger::install();

        let target = "test-dispatch-output-two-lines";
        let source = LogSource::new(target, LogOptions::default());
        dispatch_output(
            &source,
            DecodeOutput::TwoLines {
                earlier: LogLine {
                    level: LogLevel::Info,
                    text: "earlier-line".to_owned(),
                },
                current: LogLine {
                    level: LogLevel::Warn,
                    text: "current-line".to_owned(),
                },
            },
        );

        assert_eq!(
            records_for(target),
            [
                LogRecord {
                    level: Level::Info,
                    text: "earlier-line".to_owned(),
                },
                LogRecord {
                    level: Level::Warn,
                    text: "current-line".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn a_null_text_pointer_is_logged_as_a_warning() {
        CapturingLogger::install();

        let target = "test-null-text-pointer";
        let source = LogSource::new(target, LogOptions::default());
        let source_ptr = ptr::from_ref(&source).cast::<c_void>().cast_mut();
        unsafe {
            logs_to_log(
                llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_INFO,
                ptr::null(),
                source_ptr,
            );
        }

        assert_eq!(
            records_for(target),
            [LogRecord {
                level: Level::Warn,
                text: "received NULL text pointer from llama.cpp log callback".to_owned(),
            }]
        );
    }

    #[test]
    fn a_poisoned_decoder_still_decodes_logs() {
        CapturingLogger::install();

        let target = "test-decoder-mutex-poison";
        let source = LogSource::new(target, LogOptions::default());

        thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let _guard = source
                    .decoder
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let _second_guard = source.decoder.try_lock().expect("intentional poison");
            });
            let _ = handle.join();
        });

        assert!(source.decoder.is_poisoned());

        invoke_callback(
            llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_INFO,
            c"after-poison\n",
            &source,
        );

        assert_eq!(
            records_for(target),
            [LogRecord {
                level: Level::Info,
                text: "after-poison".to_owned(),
            }]
        );
    }

    #[test]
    fn the_builder_disables_logs_and_demotes_info_to_debug() {
        let options = LogOptions::default()
            .with_logs_enabled(false)
            .with_demote_info_to_debug(true);

        assert!(options.disabled);
        assert!(options.demote_info_to_debug);
    }
}
