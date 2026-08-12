use std::path::Path;

/// Runs `body` with the build-script environment the toolchain crates require,
/// returning whatever it produced.
pub fn with_cc_environment_value<TValue, TBody: FnOnce() -> TValue>(
    out_dir: &Path,
    body: TBody,
) -> TValue {
    let mut produced = None;

    with_cc_environment(out_dir, || produced = Some(body()));

    produced.expect("the body always runs")
}

/// Runs `body` with the build-script environment `cc` requires to resolve a
/// compiler and place its output.
pub fn with_cc_environment<TBody: FnOnce()>(out_dir: &Path, body: TBody) {
    let host = format!("{}-apple-darwin", std::env::consts::ARCH);

    unsafe {
        std::env::set_var("OUT_DIR", out_dir);
        std::env::set_var("HOST", &host);
        std::env::set_var("TARGET", &host);
        std::env::set_var("OPT_LEVEL", "0");
        std::env::set_var("PROFILE", "debug");
        std::env::set_var("NUM_JOBS", "1");
    }

    body();

    unsafe {
        std::env::remove_var("OUT_DIR");
        std::env::remove_var("HOST");
        std::env::remove_var("TARGET");
        std::env::remove_var("OPT_LEVEL");
        std::env::remove_var("PROFILE");
        std::env::remove_var("NUM_JOBS");
    }
}
