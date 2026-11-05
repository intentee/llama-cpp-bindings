use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::process::Command;

use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

const EMITTER: &str = env!("CARGO_BIN_EXE_llama-cpp-wrapper-sources");
const SYS_DIR: &str = "/repo/llama-cpp-bindings-sys";

fn emitter_exit_code(arguments: &[&str]) -> Option<i32> {
    Command::new(EMITTER)
        .args(arguments)
        .status()
        .expect("the emitter binary must run")
        .code()
}

fn temporary_path(suffix: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "llama-cpp-wrapper-sources-{}-{suffix}",
        process::id()
    ))
}

#[test]
fn the_binary_emits_one_entry_and_one_response_line_per_wrapper_source() {
    let database_path = temporary_path("compile_commands.json");
    let response_file_path = temporary_path("wrapper_sources.rsp");

    assert_eq!(
        emitter_exit_code(&[
            SYS_DIR,
            &database_path.display().to_string(),
            &response_file_path.display().to_string(),
        ]),
        Some(0)
    );

    let database = fs::read_to_string(&database_path).expect("the database must be written");
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&database).expect("the database must be a JSON array");

    assert_eq!(entries.len(), WRAPPER_SOURCES.len());

    let response_file =
        fs::read_to_string(&response_file_path).expect("the response file must be written");

    assert_eq!(response_file.lines().count(), WRAPPER_SOURCES.len());
    fs::remove_file(&database_path).expect("the database must be removable");
    fs::remove_file(&response_file_path).expect("the response file must be removable");
}

#[test]
fn the_binary_fails_when_the_database_cannot_be_written() {
    assert_eq!(
        emitter_exit_code(&[
            SYS_DIR,
            "/nonexistent-directory/compile_commands.json",
            "/nonexistent-directory/wrapper_sources.rsp",
        ]),
        Some(1)
    );
}

#[test]
fn the_binary_requires_every_argument() {
    let database_path = temporary_path("unwritten.json");

    assert_eq!(
        [
            emitter_exit_code(&[]),
            emitter_exit_code(&[SYS_DIR]),
            emitter_exit_code(&[SYS_DIR, &database_path.display().to_string()]),
        ],
        [Some(1), Some(1), Some(1)]
    );
}
