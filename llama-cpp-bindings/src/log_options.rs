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

extern "C" fn logs_to_trace(
    level: llama_cpp_bindings_sys::ggml_log_level,
    text: *const ::std::os::raw::c_char,
    data: *mut ::std::os::raw::c_void,
) {
    // In the "fast-path" (i.e. the vast majority of logs) we want to avoid needing to take the log state
    // lock at all. Similarly, we try to avoid any heap allocations within this function. This is accomplished
    // by being a dummy pass-through to tracing in the normal case of DEBUG/INFO/WARN/ERROR logs that are
    // newline terminated and limiting the slow-path of locks and/or heap allocations for other cases.
    use std::borrow::Borrow;

    let log_state = unsafe { &*(data as *const crate::log::State) };

    if log_state.options.disabled {
        return;
    }

    // If the log level is disabled, we can just return early
    if !log_state.is_enabled_for_level(level) {
        log_state.update_previous_level_for_disabled_log(level);

        return;
    }

    let text = unsafe { std::ffi::CStr::from_ptr(text) };
    let text = text.to_string_lossy();
    let text: &str = text.borrow();

    // As best I can tell llama.cpp / ggml require all log format strings at call sites to have the '\n'.
    // If it's missing, it means that you expect more logs via CONT (or there's a typo in the codebase). To
    // distinguish typo from intentional support for CONT, we have to buffer until the next message comes in
    // to know how to flush it.

    if level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT {
        log_state.cont_buffered_log(text);
    } else if text.ends_with('\n') {
        log_state.emit_non_cont_line(level, text);
    } else {
        log_state.buffer_non_cont(level, text);
    }
}

/// Redirect llama.cpp logs into tracing.
pub fn send_logs_to_tracing(options: LogOptions) {
    // We set up separate log states for llama.cpp and ggml to make sure that CONT logs between the two
    // can't possibly interfere with each other. In other words, if llama.cpp emits a log without a trailing
    // newline and calls a GGML function, the logs won't be weirdly intermixed and instead we'll llama.cpp logs
    // will CONT previous llama.cpp logs and GGML logs will CONT previous ggml logs.
    let llama_heap_state = Box::as_ref(crate::log::LLAMA_STATE.get_or_init(|| {
        Box::new(crate::log::State::new(
            crate::log::Module::LlamaCpp,
            options.clone(),
        ))
    })) as *const _;
    let ggml_heap_state = Box::as_ref(
        crate::log::GGML_STATE
            .get_or_init(|| Box::new(crate::log::State::new(crate::log::Module::Ggml, options))),
    ) as *const _;

    unsafe {
        // GGML has to be set after llama since setting llama sets ggml as well.
        llama_cpp_bindings_sys::llama_log_set(Some(logs_to_trace), llama_heap_state as *mut _);
        llama_cpp_bindings_sys::ggml_log_set(Some(logs_to_trace), ggml_heap_state as *mut _);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::util::SubscriberInitExt;

    use super::logs_to_trace;
    use crate::log::{Module, State};
    use crate::log_options::LogOptions;

    struct Logger {
        #[allow(unused)]
        guard: DefaultGuard,
        logs: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<String>>>);

    impl std::io::Write for VecWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let log_line = String::from_utf8(buf.to_vec()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid UTF-8")
            })?;
            self.0.lock().unwrap().push(log_line);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn create_logger(max_level: tracing::Level) -> Logger {
        let logs = Arc::new(Mutex::new(vec![]));
        let writer = VecWriter(logs.clone());

        Logger {
            guard: tracing_subscriber::fmt()
                .with_max_level(max_level)
                .with_ansi(false)
                .without_time()
                .with_file(false)
                .with_line_number(false)
                .with_level(false)
                .with_target(false)
                .with_writer(move || writer.clone())
                .finish()
                .set_default(),
            logs,
        }
    }

    #[test]
    fn cont_disabled_log() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG,
            c"Hello ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"world\n".as_ptr(),
            log_ptr,
        );

        assert!(logger.logs.lock().unwrap().is_empty());

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG,
            c"Hello ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"world".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"\n".as_ptr(),
            log_ptr,
        );
    }

    #[test]
    fn cont_enabled_log() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"Hello ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"world\n".as_ptr(),
            log_ptr,
        );

        // Not sure where the extra \n comes from.
        assert_eq!(*logger.logs.lock().unwrap(), vec!["Hello world\n\n"]);
    }

    #[test]
    fn disabled_logs_are_suppressed() {
        let logger = create_logger(tracing::Level::DEBUG);
        let disabled_options = LogOptions::default().with_logs_enabled(false);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, disabled_options));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"Should not appear\n".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR,
            c"Also suppressed\n".as_ptr(),
            log_ptr,
        );

        assert!(logger.logs.lock().unwrap().is_empty());
    }

    #[test]
    fn with_logs_enabled_true() {
        let options = LogOptions::default().with_logs_enabled(true);

        assert!(!options.disabled);
    }

    #[test]
    fn with_logs_enabled_false() {
        let options = LogOptions::default().with_logs_enabled(false);

        assert!(options.disabled);
    }

    #[test]
    fn info_level_log_emitted() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"info message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();

        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("info message"));
    }

    #[test]
    fn warn_level_log_emitted() {
        let logger = create_logger(tracing::Level::WARN);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN,
            c"warning message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();

        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("warning message"));
    }

    #[test]
    fn error_level_log_emitted() {
        let logger = create_logger(tracing::Level::ERROR);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR,
            c"error message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();

        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("error message"));
    }

    #[test]
    fn debug_level_log_emitted_when_enabled() {
        let logger = create_logger(tracing::Level::DEBUG);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG,
            c"debug message\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();

        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("debug message"));
    }

    #[test]
    fn submodule_extraction_from_log_text() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"sampling: initialized\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();

        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("initialized"));
    }

    #[test]
    fn multi_part_cont_log() {
        let logger = create_logger(tracing::Level::INFO);
        let mut log_state = Box::new(State::new(Module::LlamaCpp, LogOptions::default()));
        let log_ptr = log_state.as_mut() as *mut State as *mut std::os::raw::c_void;

        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO,
            c"part1 ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"part2 ".as_ptr(),
            log_ptr,
        );
        logs_to_trace(
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT,
            c"part3\n".as_ptr(),
            log_ptr,
        );

        let logs = logger.logs.lock().unwrap();

        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("part1 part2 part3"));
    }
}
