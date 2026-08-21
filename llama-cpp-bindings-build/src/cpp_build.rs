use std::path::PathBuf;

use crate::BuildContext;
use crate::target_os::TargetOs;

fn msvc_flags(target_os: &TargetOs) -> &'static [&'static str] {
    if target_os.is_msvc() {
        &["/std:c++17", "/EHsc"]
    } else {
        &[]
    }
}

fn links_stdlib_statically(target_os: &TargetOs) -> bool {
    target_os.is_android() && cfg!(feature = "static-stdcxx")
}

pub fn cpp_build(
    context: &BuildContext,
    include_dirs: Vec<PathBuf>,
    sources: Vec<PathBuf>,
) -> cc::Build {
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .target(&context.target_triple)
        .host(&context.host)
        .out_dir(&context.out_dir)
        .opt_level_str(&context.opt_level)
        .debug(context.debug)
        .static_crt(context.static_crt);

    for include_dir in include_dirs {
        build.include(include_dir);
    }

    build.flag_if_supported("-std=c++17").pic(true);

    for flag in msvc_flags(&context.target_os) {
        build.flag(flag);
    }

    if links_stdlib_statically(&context.target_os) {
        build.cpp_link_stdlib(None);
    }

    for source in sources {
        build.file(&source);
    }

    build
}

#[cfg(test)]
mod tests {
    use crate::host_target_triple::host_target_triple;
    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;
    use crate::test_build_context::test_build_context;

    use super::cpp_build;
    use super::links_stdlib_statically;
    use super::msvc_flags;

    #[test]
    fn platform_flag_decisions_cover_every_target() {
        let msvc = TargetOs::from_target_triple("x86_64-pc-windows-msvc").expect("msvc");
        let android = TargetOs::from_target_triple("aarch64-linux-android").expect("android");
        let apple = TargetOs::from_target_triple("aarch64-apple-darwin").expect("apple");
        let linux = TargetOs::from_target_triple("x86_64-unknown-linux-gnu").expect("linux");

        assert_eq!(msvc_flags(&msvc), &["/std:c++17", "/EHsc"]);
        assert!(msvc_flags(&apple).is_empty());
        assert!(msvc_flags(&android).is_empty());
        assert!(msvc_flags(&linux).is_empty());

        assert!(!links_stdlib_statically(&msvc));
        assert!(!links_stdlib_statically(&apple));
        assert!(!links_stdlib_statically(&linux));
        assert_eq!(
            links_stdlib_statically(&android),
            cfg!(feature = "static-stdcxx")
        );
    }

    #[test]
    fn the_builder_is_configured_for_every_target_without_compiling() {
        let scratch = ScratchDir::new("cppbuild-targets");

        for triple in [
            "x86_64-pc-windows-msvc",
            "aarch64-linux-android",
            "aarch64-apple-darwin",
            "x86_64-unknown-linux-gnu",
        ] {
            let context =
                test_build_context(scratch.path(), &scratch.path().join("llama.cpp"), triple);

            let _ = cpp_build(&context, vec![scratch.path().join("include")], Vec::new());
        }
    }

    #[test]
    fn the_builder_resolves_the_host_compiler_from_the_context() {
        let scratch = ScratchDir::new("cppbuild-host");
        let context = test_build_context(
            scratch.path(),
            &scratch.path().join("llama.cpp"),
            &host_target_triple(),
        );

        let compiler = cpp_build(&context, Vec::new(), Vec::new())
            .try_get_compiler()
            .expect("the host C++ compiler must resolve");

        assert!(compiler.is_like_msvc() || compiler.is_like_gnu() || compiler.is_like_clang());
    }
}
