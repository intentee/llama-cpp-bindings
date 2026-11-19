use std::ffi::CStr;
use std::ptr;

use log::Level;
use serial_test::serial;

use llama_cpp_backend::log_options::LogOptions;
use llama_cpp_backend_tests::registered_log_callback::RegisteredLogCallback;
use llama_cpp_capturing_logger::captured_record::CapturedRecord;
use llama_cpp_capturing_logger::capturing_logger::CapturingLogger;

fn records_with_text(target: &str, text: &str) -> Vec<CapturedRecord> {
    CapturingLogger::records_for(target)
        .into_iter()
        .filter(|CapturedRecord { message, .. }| message == text)
        .collect()
}

fn emit_warning(callback: &RegisteredLogCallback, text: &CStr) -> bool {
    callback.emit(llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_WARN, text)
}

#[test]
#[serial]
fn sending_logs_again_applies_the_new_options() {
    CapturingLogger::install();
    LogOptions::default().send_logs_to_log();
    LogOptions::default()
        .with_logs_enabled(false)
        .send_logs_to_log();

    let silenced = emit_warning(
        &RegisteredLogCallback::of_ggml(),
        c"silenced by the second call\n",
    );
    LogOptions::default().send_logs_to_log();

    assert!(silenced);
    assert_eq!(
        records_with_text("ggml", "silenced by the second call"),
        Vec::<CapturedRecord>::new()
    );
}

#[test]
#[serial]
fn sending_logs_to_log_creates_the_llama_cpp_source_and_routes_ggml_logs() {
    CapturingLogger::install();
    LogOptions::default().send_logs_to_log();

    let model_of_no_splits = unsafe {
        llama_cpp_bindings_sys::bindings::llama_model_load_from_splits(
            ptr::null_mut(),
            0,
            llama_cpp_bindings_sys::bindings::llama_model_default_params(),
        )
    };

    assert!(model_of_no_splits.is_null());
    assert!(emit_warning(
        &RegisteredLogCallback::of_ggml(),
        c"routed through ggml\n"
    ));
    assert_eq!(
        records_with_text(
            "llama.cpp",
            "llama_model_load_from_splits: list of splits is empty"
        ),
        [CapturedRecord {
            level: Level::Error,
            target: "llama.cpp".to_owned(),
            message: "llama_model_load_from_splits: list of splits is empty".to_owned(),
        }]
    );
    assert_eq!(
        records_with_text("ggml", "routed through ggml"),
        [CapturedRecord {
            level: Level::Warn,
            target: "ggml".to_owned(),
            message: "routed through ggml".to_owned(),
        }]
    );
}
