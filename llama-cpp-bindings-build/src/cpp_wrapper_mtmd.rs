use std::path::Path;

use crate::target_os::TargetOs;

pub fn compile_mtmd(llama_src: &Path, target_os: &TargetOs) {
    let mtmd_src = llama_src.join("tools/mtmd");
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .include(&mtmd_src)
        .include(llama_src)
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("common"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);

    if target_os.is_msvc() {
        build.flag("/std:c++17");
    }

    let pattern = mtmd_src.join("**/*.cpp");
    let pattern_str = pattern.to_string_lossy();

    let Ok(entries) = glob::glob(&pattern_str) else {
        println!("cargo:warning=failed to glob mtmd sources: {pattern_str}");

        return;
    };

    for entry in entries {
        match entry {
            Ok(path) => {
                let filename = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();

                if filename == "mtmd-cli.cpp" || filename == "deprecation-warning.cpp" {
                    continue;
                }

                build.file(&path);
            }
            Err(error) => println!("cargo:warning=mtmd glob error: {error}"),
        }
    }

    build.compile("mtmd");
}
