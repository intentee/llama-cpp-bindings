#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkKind {
    Unspecified,
    Static,
    Dylib,
    Framework,
}
