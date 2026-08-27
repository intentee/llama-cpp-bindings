#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeLibrary {
    pub link_kind: &'static str,
    pub name: &'static str,
}
