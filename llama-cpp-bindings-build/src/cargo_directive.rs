use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::path::PathBuf;

use crate::native_library::NativeLibrary;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CargoDirective {
    RerunIfChanged(PathBuf),
    RerunIfEnvChanged(String),
    LinkSearch(PathBuf),
    LinkSearchNative(PathBuf),
    LinkLibrary(NativeLibrary),
    Metadata { key: &'static str, value: String },
    Debug(String),
}

impl CargoDirective {
    #[must_use]
    pub const fn is_debug(&self) -> bool {
        matches!(self, Self::Debug(_))
    }
}

impl Display for CargoDirective {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::RerunIfChanged(path) => {
                write!(formatter, "cargo:rerun-if-changed={}", path.display())
            }
            Self::RerunIfEnvChanged(name) => write!(formatter, "cargo:rerun-if-env-changed={name}"),
            Self::LinkSearch(path) => {
                write!(formatter, "cargo:rustc-link-search={}", path.display())
            }
            Self::LinkSearchNative(path) => {
                write!(
                    formatter,
                    "cargo:rustc-link-search=native={}",
                    path.display()
                )
            }
            Self::LinkLibrary(library) => write!(formatter, "cargo:rustc-link-lib={library}"),
            Self::Metadata { key, value } => write!(formatter, "cargo:{key}={value}"),
            Self::Debug(message) => write!(formatter, "cargo:warning=[DEBUG] {message}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::CargoDirective;
    use crate::link_kind::LinkKind;
    use crate::native_library::NativeLibrary;

    fn every_directive() -> [CargoDirective; 7] {
        [
            CargoDirective::RerunIfChanged(PathBuf::from("wrapper.h")),
            CargoDirective::RerunIfEnvChanged("CUDAARCHS".to_owned()),
            CargoDirective::LinkSearch(PathBuf::from("/out/lib")),
            CargoDirective::LinkSearchNative(PathBuf::from("/out/build/common")),
            CargoDirective::LinkLibrary(NativeLibrary {
                link_kind: LinkKind::Static,
                name: "llama",
            }),
            CargoDirective::Metadata {
                key: "backends_dir",
                value: "/out/backends".to_owned(),
            },
            CargoDirective::Debug("TARGET: x86_64-unknown-linux-gnu".to_owned()),
        ]
    }

    #[test]
    fn every_directive_is_written_in_the_cargo_syntax() {
        assert_eq!(
            every_directive().map(|directive| directive.to_string()),
            [
                "cargo:rerun-if-changed=wrapper.h",
                "cargo:rerun-if-env-changed=CUDAARCHS",
                "cargo:rustc-link-search=/out/lib",
                "cargo:rustc-link-search=native=/out/build/common",
                "cargo:rustc-link-lib=static=llama",
                "cargo:backends_dir=/out/backends",
                "cargo:warning=[DEBUG] TARGET: x86_64-unknown-linux-gnu",
            ]
        );
    }

    #[test]
    fn only_debug_messages_are_debug_directives() {
        assert_eq!(
            every_directive().map(|directive| directive.is_debug()),
            [false, false, false, false, false, false, true]
        );
    }
}
