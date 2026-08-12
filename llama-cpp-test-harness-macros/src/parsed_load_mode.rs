use syn::Expr;
use syn::ExprPath;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParsedLoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl ParsedLoadMode {
    pub const fn variant_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::None => "None",
            Self::Mmap => "Mmap",
            Self::Mlock => "Mlock",
            Self::MmapMlock => "MmapMlock",
            Self::DirectIo => "DirectIo",
        }
    }

    pub fn parse(expression: &Expr, field: &str) -> syn::Result<Self> {
        let Expr::Path(ExprPath { path, .. }) = expression else {
            return Err(syn::Error::new_spanned(
                expression,
                format!(
                    "field `{field}` expects a bare variant name such as `Auto`, `Mmap`, or `MmapMlock`"
                ),
            ));
        };
        let variant_ident = path.get_ident().ok_or_else(|| {
            syn::Error::new_spanned(
                path,
                format!("field `{field}` expects the bare variant name, not a qualified path"),
            )
        })?;

        match variant_ident.to_string().as_str() {
            "Auto" => Ok(Self::Auto),
            "None" => Ok(Self::None),
            "Mmap" => Ok(Self::Mmap),
            "Mlock" => Ok(Self::Mlock),
            "MmapMlock" => Ok(Self::MmapMlock),
            "DirectIo" => Ok(Self::DirectIo),
            other => Err(syn::Error::new_spanned(
                variant_ident,
                format!(
                    "unknown load mode `{other}`; expected one of: Auto, None, Mmap, Mlock, MmapMlock, DirectIo"
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use syn::parse_str;

    use super::ParsedLoadMode;

    fn parse(source: &str) -> syn::Result<ParsedLoadMode> {
        let expression: syn::Expr = parse_str(source)?;
        ParsedLoadMode::parse(&expression, "load_mode")
    }

    #[test]
    fn parses_every_known_variant() {
        for (source, expected) in [
            ("Auto", ParsedLoadMode::Auto),
            ("None", ParsedLoadMode::None),
            ("Mmap", ParsedLoadMode::Mmap),
            ("Mlock", ParsedLoadMode::Mlock),
            ("MmapMlock", ParsedLoadMode::MmapMlock),
            ("DirectIo", ParsedLoadMode::DirectIo),
        ] {
            assert_eq!(parse(source).expect("valid variant"), expected);
        }
    }

    #[test]
    fn variant_name_matches_the_parsed_source() {
        for source in ["Auto", "None", "Mmap", "Mlock", "MmapMlock", "DirectIo"] {
            assert_eq!(parse(source).expect("valid variant").variant_name(), source);
        }
    }

    #[test]
    fn unknown_variant_is_rejected() {
        let message = parse("Mystery")
            .expect_err("unknown variant must fail")
            .to_string();

        assert!(message.contains("unknown load mode"), "got: {message}");
    }

    #[test]
    fn non_path_expression_is_rejected() {
        let message = parse("\"Auto\"")
            .expect_err("string literal must fail")
            .to_string();

        assert!(message.contains("bare variant name"), "got: {message}");
    }

    #[test]
    fn qualified_path_variant_is_rejected() {
        let message = parse("LlamaLoadMode::Auto")
            .expect_err("qualified path must fail")
            .to_string();

        assert!(message.contains("not a qualified path"), "got: {message}");
    }

    #[test]
    fn unparseable_input_returns_err() {
        let result = parse("@&^!");

        assert!(
            result.is_err(),
            "garbage input must fail to parse as syn::Expr"
        );
    }
}
