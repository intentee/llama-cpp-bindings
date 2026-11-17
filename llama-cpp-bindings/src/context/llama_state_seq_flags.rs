use std::ops::BitOr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaStateSeqFlags {
    pub flags: u32,
}

impl LlamaStateSeqFlags {
    #[must_use]
    pub const fn empty() -> Self {
        Self { flags: 0 }
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.flags
    }

    #[must_use]
    pub const fn contains(self, Self { flags }: Self) -> bool {
        (self.flags & flags) == flags
    }
}

impl BitOr for LlamaStateSeqFlags {
    type Output = Self;

    fn bitor(self, Self { flags }: Self) -> Self {
        Self {
            flags: self.flags | flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaStateSeqFlags;

    #[test]
    fn empty_has_no_bits_set() {
        assert_eq!(LlamaStateSeqFlags::empty().bits(), 0);
    }

    #[test]
    fn bitor_combines_flags() {
        let combined = LlamaStateSeqFlags::empty() | LlamaStateSeqFlags { flags: 1 };

        assert_eq!(combined.bits(), 1);
    }

    #[test]
    fn contains_detects_set_flag() {
        let flags = LlamaStateSeqFlags { flags: 1 };

        assert!(flags.contains(LlamaStateSeqFlags { flags: 1 }));
    }

    #[test]
    fn empty_does_not_contain_partial_only() {
        let flags = LlamaStateSeqFlags::empty();

        assert!(!flags.contains(LlamaStateSeqFlags { flags: 1 }));
    }
}
