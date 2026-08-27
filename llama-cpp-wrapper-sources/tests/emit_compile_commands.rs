use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

const EMITTER: &str = env!("CARGO_BIN_EXE_llama-cpp-wrapper-sources");
const SYS_DIR: &str = "/repo/llama-cpp-bindings-sys";

fn emitter_succeeded(arguments: &[&str]) -> bool {
    Command::new(EMITTER)
        .args(arguments)
        .status()
        .is_ok_and(|status| status.success())
}

fn temporary_path(suffix: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "llama-cpp-wrapper-sources-{}-{suffix}",
        std::process::id()
    ))
}

#[test]
fn the_binary_emits_one_entry_and_one_response_line_per_wrapper_source() {
    let database_path = temporary_path("compile_commands.json");
    let response_file_path = temporary_path("wrapper_sources.rsp");

    assert!(emitter_succeeded(&[
        SYS_DIR,
        &database_path.display().to_string(),
        &response_file_path.display().to_string(),
    ]));

    let database = fs::read_to_string(&database_path).unwrap_or_default();
    let entries: Vec<serde_json::Value> = serde_json::from_str(&database).unwrap_or_default();

    assert_eq!(entries.len(), WRAPPER_SOURCES.len());

    let response_file = fs::read_to_string(&response_file_path).unwrap_or_default();

    assert_eq!(response_file.lines().count(), WRAPPER_SOURCES.len());
    assert!(fs::remove_file(&database_path).is_ok());
    assert!(fs::remove_file(&response_file_path).is_ok());
}

#[test]
fn the_binary_fails_when_the_database_cannot_be_written() {
    assert!(!emitter_succeeded(&[
        SYS_DIR,
        "/nonexistent-directory/compile_commands.json",
        "/nonexistent-directory/wrapper_sources.rsp",
    ]));
}

#[test]
fn the_binary_requires_every_output_path() {
    let database_path = temporary_path("unwritten.json");

    assert!(!emitter_succeeded(&[]));
    assert!(!emitter_succeeded(&[SYS_DIR]));
    assert!(!emitter_succeeded(&[
        SYS_DIR,
        &database_path.display().to_string()
    ]));
}
