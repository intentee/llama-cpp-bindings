mod host_triple_error;

use std::env;
use std::fs;
use std::path::PathBuf;

use crate::host_triple_error::HostTripleError;

fn main() -> Result<(), HostTripleError> {
    println!("cargo:rerun-if-changed=build.rs");

    env::var("TARGET")
        .map_err(HostTripleError::TargetUnreadable)
        .and_then(|target| {
            env::var("OUT_DIR")
                .map_err(HostTripleError::OutDirUnreadable)
                .and_then(|out_dir| {
                    fs::write(PathBuf::from(out_dir).join("host_triple"), target)
                        .map_err(HostTripleError::HostTripleUnwritable)
                })
        })
}
