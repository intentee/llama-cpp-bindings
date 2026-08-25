use std::env;
use std::path::Path;

use crate::BuildError;
use crate::apple_variant::AppleVariant;
use crate::debug_log;
use crate::native_library::NativeLibrary;
use crate::target_os::TargetOs;
use crate::windows_variant::WindowsVariant;

pub fn link_libraries(
    cmake_dir: &Path,
    build_dir: &Path,
    target_os: TargetOs,
    cargo_cfg_target_env: &str,
    build_shared_libs: bool,
    profile: &str,
) -> Result<(), BuildError> {
    emit_search_paths(cmake_dir, build_dir);
    link_system_ggml_paths()?;
    link_cmake_built_libraries(cmake_dir, build_shared_libs, profile);
    link_cuda_libraries(target_os, build_shared_libs);
    link_rocm_libraries(build_shared_libs)?;
    link_openmp(cargo_cfg_target_env);
    link_platform_system_libraries(target_os);

    Ok(())
}

fn emit_search_paths(cmake_dir: &Path, build_dir: &Path) {
    println!(
        "cargo:rustc-link-search={}",
        cmake_dir.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search={}",
        cmake_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());
}

fn link_system_ggml_paths() -> Result<(), BuildError> {
    if !cfg!(feature = "system-ggml") {
        return Ok(());
    }

    println!("cargo:rerun-if-env-changed=GGML_LIBRARY_DIR");
    let library_dir = env::var("GGML_LIBRARY_DIR").map_err(|source| BuildError::Environment {
        name: "GGML_LIBRARY_DIR",
        source,
    })?;
    let library_dir = Path::new(&library_dir);
    if !library_dir.is_dir() {
        return Err(BuildError::EnvironmentDirectory {
            name: "GGML_LIBRARY_DIR",
            path: library_dir.to_path_buf(),
        });
    }
    println!("cargo:rustc-link-search=native={}", library_dir.display());

    Ok(())
}

fn link_cmake_built_libraries(cmake_dir: &Path, build_shared_libs: bool, profile: &str) {
    emit_private_dependency_search_paths(cmake_dir, profile);
    for NativeLibrary { link_kind, name } in native_libraries(build_shared_libs) {
        let link = format!("cargo:rustc-link-lib={link_kind}={name}");
        debug_log!("LINK {link}");
        println!("{link}");
    }
}

fn native_libraries(build_shared_libs: bool) -> Vec<NativeLibrary> {
    let cmake_kind = if build_shared_libs { "dylib" } else { "static" };
    let mut libraries = vec![NativeLibrary {
        link_kind: cmake_kind,
        name: "llama-common",
    }];

    if !build_shared_libs {
        libraries.extend([
            NativeLibrary {
                link_kind: "static",
                name: "llama-common-base",
            },
            NativeLibrary {
                link_kind: "static",
                name: "cpp-httplib",
            },
        ]);
    }

    libraries.push(NativeLibrary {
        link_kind: cmake_kind,
        name: "mtmd",
    });
    if !build_shared_libs {
        libraries.push(NativeLibrary {
            link_kind: "static",
            name: "vendor-hash",
        });
    }
    libraries.push(NativeLibrary {
        link_kind: cmake_kind,
        name: "llama",
    });

    if cfg!(feature = "system-ggml") {
        let ggml_kind = if cfg!(feature = "system-ggml-static") {
            "static"
        } else {
            "dylib"
        };
        libraries.extend([
            NativeLibrary {
                link_kind: ggml_kind,
                name: "ggml-cpu",
            },
            NativeLibrary {
                link_kind: ggml_kind,
                name: "ggml-base",
            },
            NativeLibrary {
                link_kind: ggml_kind,
                name: "ggml",
            },
        ]);
        return libraries;
    }

    if !cfg!(feature = "dynamic-backends") {
        if cfg!(feature = "cuda") {
            libraries.push(NativeLibrary {
                link_kind: cmake_kind,
                name: "ggml-cuda",
            });
        }
        if cfg!(feature = "metal") {
            libraries.push(NativeLibrary {
                link_kind: cmake_kind,
                name: "ggml-metal",
            });
        }
        if cfg!(feature = "vulkan") {
            libraries.push(NativeLibrary {
                link_kind: cmake_kind,
                name: "ggml-vulkan",
            });
        }
        if cfg!(feature = "rocm") {
            libraries.push(NativeLibrary {
                link_kind: cmake_kind,
                name: "ggml-hip",
            });
        }
    }

    libraries.extend([
        NativeLibrary {
            link_kind: cmake_kind,
            name: "ggml-cpu",
        },
        NativeLibrary {
            link_kind: cmake_kind,
            name: "ggml-base",
        },
        NativeLibrary {
            link_kind: cmake_kind,
            name: "ggml",
        },
    ]);
    libraries
}

fn emit_private_dependency_search_paths(cmake_dir: &Path, profile: &str) {
    let common_lib_dir = cmake_dir.join("build").join("common");
    let httplib_dir = cmake_dir.join("build").join("vendor").join("cpp-httplib");
    let hash_dir = cmake_dir.join("build").join("vendor").join("hash");

    emit_search_path_with_profile(&common_lib_dir, profile);
    emit_search_path_with_profile(&httplib_dir, profile);
    emit_search_path_with_profile(&hash_dir, profile);
}

fn emit_search_path_with_profile(lib_dir: &Path, profile: &str) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    let profile_dir = lib_dir.join(profile);

    println!("cargo:rustc-link-search=native={}", profile_dir.display());
}

fn link_cuda_libraries(target_os: TargetOs, build_shared_libs: bool) {
    if !cfg!(feature = "cuda") || build_shared_libs {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    for lib_dir in find_cuda_helper::find_cuda_lib_dirs() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    match target_os {
        TargetOs::Windows(_) => link_cuda_windows(),
        _ => link_cuda_unix(),
    }
}

fn link_cuda_windows() {
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");

    if !cfg!(feature = "cuda-no-vmm") {
        println!("cargo:rustc-link-lib=cuda");
    }
}

fn link_cuda_unix() {
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=cublas_static");
    println!("cargo:rustc-link-lib=static=cublasLt_static");

    if !cfg!(feature = "cuda-no-vmm") {
        println!("cargo:rustc-link-lib=cuda");
    }

    println!("cargo:rustc-link-lib=static=culibos");
}

fn link_rocm_libraries(build_shared_libs: bool) -> Result<(), BuildError> {
    if !cfg!(feature = "rocm") || build_shared_libs {
        return Ok(());
    }

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    let rocm_path = env::var("ROCM_PATH").map_err(|source| BuildError::Environment {
        name: "ROCM_PATH",
        source,
    })?;

    let rocm_lib = Path::new(&rocm_path).join("lib");

    if !rocm_lib.is_dir() {
        return Err(BuildError::EnvironmentDirectory {
            name: "ROCM_PATH",
            path: rocm_lib,
        });
    }

    println!("cargo:rustc-link-search=native={}", rocm_lib.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=hipblas");

    Ok(())
}

fn link_openmp(cargo_cfg_target_env: &str) {
    if cfg!(feature = "openmp") && cargo_cfg_target_env == "gnu" {
        println!("cargo:rustc-link-lib=gomp");
    }
}

fn link_platform_system_libraries(target_os: TargetOs) {
    match target_os {
        TargetOs::Windows(WindowsVariant::Msvc) => {
            println!("cargo:rustc-link-lib=advapi32");
        }
        TargetOs::Linux => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        TargetOs::Apple(variant) => {
            link_apple_frameworks(variant);
        }
        TargetOs::Android => {
            link_android_cpp_stdlib();
        }
        TargetOs::Windows(WindowsVariant::Other) => {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
}

fn link_android_cpp_stdlib() {
    if cfg!(feature = "static-stdcxx") {
        println!("cargo:rustc-link-lib=c++_static");
        println!("cargo:rustc-link-lib=c++abi");
    } else if cfg!(feature = "shared-stdcxx") {
        println!("cargo:rustc-link-lib=c++_shared");
    }
}

fn link_apple_frameworks(_variant: AppleVariant) {
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=c++");
}

#[cfg(test)]
mod native_link_graph_tests {
    use super::NativeLibrary;
    use super::native_libraries;

    #[test]
    fn the_cuda_backend_archive_is_linked_only_when_the_feature_is_enabled() {
        let links_cuda_backend = native_libraries(false)
            .iter()
            .any(|library| library.name == "ggml-cuda");

        assert_eq!(links_cuda_backend, cfg!(feature = "cuda"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn static_cuda_link_graph_contains_every_owned_archive_in_dependency_order() {
        assert_eq!(
            native_libraries(false),
            vec![
                NativeLibrary {
                    link_kind: "static",
                    name: "llama-common",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "llama-common-base",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "cpp-httplib",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "mtmd",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "vendor-hash",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "llama",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "ggml-cuda",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "ggml-cpu",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "ggml-base",
                },
                NativeLibrary {
                    link_kind: "static",
                    name: "ggml",
                },
            ]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dynamic_cuda_link_graph_uses_shared_top_level_libraries() {
        assert_eq!(
            native_libraries(true),
            vec![
                NativeLibrary {
                    link_kind: "dylib",
                    name: "llama-common",
                },
                NativeLibrary {
                    link_kind: "dylib",
                    name: "mtmd",
                },
                NativeLibrary {
                    link_kind: "dylib",
                    name: "llama",
                },
                NativeLibrary {
                    link_kind: "dylib",
                    name: "ggml-cuda",
                },
                NativeLibrary {
                    link_kind: "dylib",
                    name: "ggml-cpu",
                },
                NativeLibrary {
                    link_kind: "dylib",
                    name: "ggml-base",
                },
                NativeLibrary {
                    link_kind: "dylib",
                    name: "ggml",
                },
            ]
        );
    }
}
