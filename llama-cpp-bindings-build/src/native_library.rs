use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use crate::link_kind::LinkKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeLibrary {
    pub link_kind: LinkKind,
    pub name: &'static str,
}

impl Display for NativeLibrary {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let Self { link_kind, name } = self;

        match link_kind {
            LinkKind::Unspecified => write!(formatter, "{name}"),
            LinkKind::Static => write!(formatter, "static={name}"),
            LinkKind::Dylib => write!(formatter, "dylib={name}"),
            LinkKind::Framework => write!(formatter, "framework={name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NativeLibrary;
    use crate::link_kind::LinkKind;

    #[test]
    fn every_link_kind_prefixes_the_library_the_way_cargo_expects() {
        assert_eq!(
            [
                LinkKind::Unspecified,
                LinkKind::Static,
                LinkKind::Dylib,
                LinkKind::Framework
            ]
            .map(|link_kind| NativeLibrary {
                link_kind,
                name: "llama"
            }
            .to_string()),
            ["llama", "static=llama", "dylib=llama", "framework=llama"]
        );
    }
}
