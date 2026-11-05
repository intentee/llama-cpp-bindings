use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use llama_cpp_wrapper_sources::wrapper_headers::WRAPPER_HEADERS;
use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

const SYS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../llama-cpp-bindings-sys");

fn sys_root_files_with_extension(extension: &str) -> BTreeSet<String> {
    fs::read_dir(Path::new(SYS_DIR))
        .expect("the sys crate directory must be readable")
        .map(|entry| {
            entry
                .expect("a sys crate directory entry must be readable")
                .path()
        })
        .filter(|path| path.is_file() && path.extension() == Some(OsStr::new(extension)))
        .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
        .collect()
}

fn listed(files: &[&str]) -> BTreeSet<String> {
    files.iter().map(|file| (*file).to_owned()).collect()
}

#[test]
fn every_cpp_source_in_the_sys_crate_root_is_a_listed_wrapper_source() {
    assert_eq!(
        sys_root_files_with_extension("cpp"),
        listed(WRAPPER_SOURCES)
    );
}

#[test]
fn every_header_in_the_sys_crate_root_is_a_listed_wrapper_header() {
    assert_eq!(sys_root_files_with_extension("h"), listed(WRAPPER_HEADERS));
}
