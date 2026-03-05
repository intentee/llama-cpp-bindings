use crate::log_options::LogOptions;
use std::sync::OnceLock;
use tracing_core::{Interest, Kind, Metadata, callsite, field, identify_callsite};

static FIELD_NAMES: &[&str] = &["message", "module"];

struct OverridableFields {
    message: tracing::field::Field,
    target: tracing::field::Field,
}

macro_rules! log_cs {
    ($level:expr, $cs:ident, $meta:ident, $fields:ident, $ty:ident) => {
        struct $ty;
        static $cs: $ty = $ty;
        static $meta: Metadata<'static> = Metadata::new(
            "log event",
            "llama-cpp-bindings",
            $level,
            ::core::option::Option::None,
            ::core::option::Option::None,
            ::core::option::Option::None,
            field::FieldSet::new(FIELD_NAMES, identify_callsite!(&$cs)),
            Kind::EVENT,
        );
        static $fields: std::sync::LazyLock<OverridableFields> = std::sync::LazyLock::new(|| {
            let fields = $meta.fields();
            OverridableFields {
                message: fields
                    .field("message")
                    .expect("message field defined in FIELD_NAMES"),
                target: fields
                    .field("module")
                    .expect("module field defined in FIELD_NAMES"),
            }
        });

        impl callsite::Callsite for $ty {
            fn set_interest(&self, _: Interest) {}
            fn metadata(&self) -> &'static Metadata<'static> {
                &$meta
            }
        }
    };
}
log_cs!(
    tracing_core::Level::DEBUG,
    DEBUG_CS,
    DEBUG_META,
    DEBUG_FIELDS,
    DebugCallsite
);
log_cs!(
    tracing_core::Level::INFO,
    INFO_CS,
    INFO_META,
    INFO_FIELDS,
    InfoCallsite
);
log_cs!(
    tracing_core::Level::WARN,
    WARN_CS,
    WARN_META,
    WARN_FIELDS,
    WarnCallsite
);
log_cs!(
    tracing_core::Level::ERROR,
    ERROR_CS,
    ERROR_META,
    ERROR_FIELDS,
    ErrorCallsite
);

#[derive(Clone, Copy)]
pub enum Module {
    Ggml,
    LlamaCpp,
}

impl Module {
    const fn name(self) -> &'static str {
        match self {
            Self::Ggml => "ggml",
            Self::LlamaCpp => "llama.cpp",
        }
    }
}

fn meta_for_level(
    level: llama_cpp_bindings_sys::ggml_log_level,
) -> (&'static Metadata<'static>, &'static OverridableFields) {
    match level {
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG => (&DEBUG_META, &DEBUG_FIELDS),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO => (&INFO_META, &INFO_FIELDS),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN => (&WARN_META, &WARN_FIELDS),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR => (&ERROR_META, &ERROR_FIELDS),
        _ => {
            unreachable!("Illegal log level to be called here")
        }
    }
}

pub struct State {
    pub options: LogOptions,
    module: Module,
    buffered: std::sync::Mutex<Option<(llama_cpp_bindings_sys::ggml_log_level, String)>>,
    previous_level: std::sync::atomic::AtomicI32,
    is_buffering: std::sync::atomic::AtomicBool,
}

impl State {
    pub fn new(module: Module, options: LogOptions) -> Self {
        Self {
            options,
            module,
            buffered: std::sync::Mutex::default(),
            previous_level: std::sync::atomic::AtomicI32::default(),
            is_buffering: std::sync::atomic::AtomicBool::default(),
        }
    }

    fn generate_log(target: Module, level: llama_cpp_bindings_sys::ggml_log_level, text: &str) {
        // Annoying but tracing requires that the provided target name is a string literal and
        // even &'static str isn't enough so we have to duplicate the generation AND we can't even
        // extract the interrior module within llama.cpp/ggml to be able to propagate it forward.
        // This happens because the target is part of a static variable injected by the macro that's
        // initialized with said target.

        let (module, text) = text
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_lowercase() || *c == '_')
            .last()
            .and_then(|(pos, _)| {
                let next_two = text.get(pos + 1..pos + 3);
                if next_two == Some(": ") {
                    let (sub_module, text) = text.split_at(pos + 1);
                    let text = text.split_at(2).1;
                    Some((Some(format!("{}::{sub_module}", target.name())), text))
                } else {
                    None
                }
            })
            .unwrap_or((None, text));

        let (meta, fields) = meta_for_level(level);

        tracing::dispatcher::get_default(|dispatcher| {
            dispatcher.event(&tracing::Event::new(
                meta,
                &meta.fields().value_set(&[
                    (&fields.message, Some(&text as &dyn tracing::field::Value)),
                    (
                        &fields.target,
                        module.as_ref().map(|s| s as &dyn tracing::field::Value),
                    ),
                ]),
            ));
        });
    }

    /// Append more text to the previously buffered log. The text may or may not end with a newline.
    pub fn cont_buffered_log(&self, text: &str) {
        let mut lock = self.buffered.lock().unwrap();

        if let Some((previous_log_level, mut buffer)) = lock.take() {
            buffer.push_str(text);
            if buffer.ends_with('\n') {
                self.is_buffering
                    .store(false, std::sync::atomic::Ordering::Release);
                Self::generate_log(self.module, previous_log_level, buffer.as_str());
            } else {
                *lock = Some((previous_log_level, buffer));
            }
        } else {
            let level = self
                .previous_level
                .load(std::sync::atomic::Ordering::Acquire)
                as llama_cpp_bindings_sys::ggml_log_level;
            tracing::warn!(
                inferred_level = level,
                text = text,
                origin = "crate",
                "llama.cpp sent out a CONT log without any previously buffered message"
            );
            *lock = Some((level, text.to_string()));
        }
    }

    /// Start buffering a message. Not the CONT log level and text is missing a newline.
    pub fn buffer_non_cont(&self, level: llama_cpp_bindings_sys::ggml_log_level, text: &str) {
        debug_assert!(!text.ends_with('\n'));
        debug_assert_ne!(level, llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT);

        if let Some((previous_log_level, buffer)) = self
            .buffered
            .lock()
            .unwrap()
            .replace((level, text.to_string()))
        {
            tracing::warn!(
                level = previous_log_level,
                text = &buffer,
                origin = "crate",
                "Message buffered unnecessarily due to missing newline and not followed by a CONT"
            );
            Self::generate_log(self.module, previous_log_level, buffer.as_str());
        }

        self.is_buffering
            .store(true, std::sync::atomic::Ordering::Release);
        self.previous_level
            .store(level as i32, std::sync::atomic::Ordering::Release);
    }

    // Emit a normal unbuffered log message (not the CONT log level and the text ends with a newline).
    pub fn emit_non_cont_line(&self, level: llama_cpp_bindings_sys::ggml_log_level, text: &str) {
        debug_assert!(text.ends_with('\n'));
        debug_assert_ne!(level, llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT);

        if self
            .is_buffering
            .swap(false, std::sync::atomic::Ordering::Acquire)
            && let Some((buf_level, buf_text)) = self.buffered.lock().unwrap().take()
        {
            // This warning indicates a bug within llama.cpp
            tracing::warn!(
                level = buf_level,
                text = buf_text,
                origin = "crate",
                "llama.cpp message buffered spuriously due to missing \\n and being followed by a non-CONT message!"
            );
            Self::generate_log(self.module, buf_level, buf_text.as_str());
        }

        self.previous_level
            .store(level as i32, std::sync::atomic::Ordering::Release);

        let (text, newline) = text.split_at(text.len() - 1);
        debug_assert_eq!(newline, "\n");

        match level {
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE => {
                tracing::info!(no_log_level = true, text);
            }
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_DEBUG
            | llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO
            | llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN
            | llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR => {
                Self::generate_log(self.module, level, text)
            }
            llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT => unreachable!(),
            _ => {
                tracing::warn!(
                    level = level,
                    text = text,
                    origin = "crate",
                    "Unknown llama.cpp log level"
                );
            }
        }
    }

    pub fn update_previous_level_for_disabled_log(
        &self,
        level: llama_cpp_bindings_sys::ggml_log_level,
    ) {
        if level != llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT {
            self.previous_level
                .store(level as i32, std::sync::atomic::Ordering::Release);
        }
    }

    /// Checks whether the given log level is enabled by the current tracing subscriber.
    pub fn is_enabled_for_level(&self, level: llama_cpp_bindings_sys::ggml_log_level) -> bool {
        // CONT logs do not need to check if they are enabled.
        let level = if level == llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT {
            self.previous_level
                .load(std::sync::atomic::Ordering::Relaxed)
                as llama_cpp_bindings_sys::ggml_log_level
        } else {
            level
        };
        let (meta, _) = meta_for_level(level);
        tracing::dispatcher::get_default(|dispatcher| dispatcher.enabled(meta))
    }
}

pub static LLAMA_STATE: OnceLock<Box<State>> = OnceLock::new();
pub static GGML_STATE: OnceLock<Box<State>> = OnceLock::new();

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

    let log_state = unsafe { &*(data as *const State) };

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
    let llama_heap_state = Box::as_ref(
        LLAMA_STATE.get_or_init(|| Box::new(State::new(Module::LlamaCpp, options.clone()))),
    ) as *const _;
    let ggml_heap_state =
        Box::as_ref(GGML_STATE.get_or_init(|| Box::new(State::new(Module::Ggml, options))))
            as *const _;

    unsafe {
        // GGML has to be set after llama since setting llama sets ggml as well.
        llama_cpp_bindings_sys::llama_log_set(Some(logs_to_trace), llama_heap_state as *mut _);
        llama_cpp_bindings_sys::ggml_log_set(Some(logs_to_trace), ggml_heap_state as *mut _);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::util::SubscriberInitExt;

    use super::{Module, State, logs_to_trace};
    use crate::log_options::LogOptions;

    #[test]
    fn module_name_ggml() {
        assert_eq!(Module::Ggml.name(), "ggml");
    }

    #[test]
    fn module_name_llama_cpp() {
        assert_eq!(Module::LlamaCpp.name(), "llama.cpp");
    }

    #[test]
    fn state_new_creates_empty_buffer() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());
        let buffer = state.buffered.lock().unwrap_or_else(|err| err.into_inner());

        assert!(buffer.is_none());
        assert!(!state.options.disabled);
    }

    #[test]
    fn update_previous_level_for_disabled_log_stores_level() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN);

        let stored = state
            .previous_level
            .load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(stored, llama_cpp_bindings_sys::GGML_LOG_LEVEL_WARN as i32);
    }

    #[test]
    fn update_previous_level_ignores_cont() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR);
        state.update_previous_level_for_disabled_log(llama_cpp_bindings_sys::GGML_LOG_LEVEL_CONT);

        let stored = state
            .previous_level
            .load(std::sync::atomic::Ordering::Relaxed);

        assert_eq!(stored, llama_cpp_bindings_sys::GGML_LOG_LEVEL_ERROR as i32);
    }

    #[test]
    fn buffer_non_cont_sets_buffering_flag() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO, "partial");

        assert!(
            state
                .is_buffering
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        let buffer = state.buffered.lock().unwrap_or_else(|err| err.into_inner());

        assert!(buffer.is_some());
        let (level, text) = buffer.as_ref().unwrap();

        assert_eq!(*level, llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO);
        assert_eq!(text, "partial");
    }

    #[test]
    fn cont_buffered_log_appends_to_existing_buffer() {
        let state = State::new(Module::LlamaCpp, LogOptions::default());

        state.buffer_non_cont(llama_cpp_bindings_sys::GGML_LOG_LEVEL_INFO, "hello ");

        state.cont_buffered_log("world");

        let buffer = state.buffered.lock().unwrap_or_else(|err| err.into_inner());

        assert!(buffer.is_some());
        let (_, text) = buffer.as_ref().unwrap();

        assert_eq!(text, "hello world");
    }

    struct Logger {
        #[allow(unused)]
        guard: tracing::subscriber::DefaultGuard,
        logs: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<String>>>);

    impl std::io::Write for VecWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let log_line = String::from_utf8(buf.to_vec()).map_err(|utf8_error| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, utf8_error)
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
