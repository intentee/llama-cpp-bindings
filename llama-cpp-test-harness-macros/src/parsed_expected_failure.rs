use proc_macro2::Ident;
use syn::Expr;
use syn::ExprCall;
use syn::ExprLit;
use syn::ExprPath;
use syn::Lit;

const USAGE: &str = "field `expected_failure` expects `SetupFailure(kind, \"reason\")` or \
                     `TestFailure(\"reason\")`";

fn reason_of(expression: &Expr) -> syn::Result<String> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(reason),
        ..
    }) = expression
        && !reason.value().is_empty()
    {
        Ok(reason.value())
    } else {
        Err(syn::Error::new_spanned(
            expression,
            "`expected_failure` expects a non-empty string stating why the trial fails on purpose",
        ))
    }
}

fn setup_failure_kind_of(expression: &Expr) -> syn::Result<Ident> {
    if let Expr::Path(ExprPath { path, .. }) = expression
        && let Some(kind) = path.get_ident()
    {
        Ok(kind.clone())
    } else {
        Err(syn::Error::new_spanned(
            expression,
            "`SetupFailure` expects the bare name of a phase setup failure kind, such as \
             `ModelDownload`",
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParsedExpectedFailure {
    PhaseSetup { kind: Ident, reason: String },
    Test { reason: String },
}

impl ParsedExpectedFailure {
    pub fn parse(expression: &Expr) -> syn::Result<Self> {
        let Expr::Call(ExprCall { func, args, .. }) = expression else {
            return Err(syn::Error::new_spanned(expression, USAGE));
        };
        let arguments: Vec<&Expr> = args.iter().collect();

        match (func.as_ref(), arguments.as_slice()) {
            (Expr::Path(ExprPath { path, .. }), [kind, reason])
                if path.is_ident("SetupFailure") =>
            {
                Ok(Self::PhaseSetup {
                    kind: setup_failure_kind_of(kind)?,
                    reason: reason_of(reason)?,
                })
            }
            (Expr::Path(ExprPath { path, .. }), [reason]) if path.is_ident("TestFailure") => {
                Ok(Self::Test {
                    reason: reason_of(reason)?,
                })
            }
            _ => Err(syn::Error::new_spanned(expression, USAGE)),
        }
    }
}

#[cfg(test)]
mod tests {
    use proc_macro2::Ident;
    use proc_macro2::Span;
    use syn::parse_str;

    use super::ParsedExpectedFailure;

    fn parse(source: &str) -> Result<ParsedExpectedFailure, String> {
        ParsedExpectedFailure::parse(&parse_str(source).expect("the source must be an expression"))
            .map_err(|error| error.to_string())
    }

    #[test]
    fn a_setup_failure_names_its_kind_and_reason() {
        assert_eq!(
            parse("SetupFailure(ModelDownload, \"the repository does not exist\")"),
            Ok(ParsedExpectedFailure::PhaseSetup {
                kind: Ident::new("ModelDownload", Span::call_site()),
                reason: "the repository does not exist".to_owned(),
            })
        );
    }

    #[test]
    fn a_test_failure_names_its_reason() {
        assert_eq!(
            parse("TestFailure(\"the body bails on purpose\")"),
            Ok(ParsedExpectedFailure::Test {
                reason: "the body bails on purpose".to_owned(),
            })
        );
    }

    #[test]
    fn a_failure_that_is_not_a_call_is_refused() {
        assert_eq!(
            parse("ModelDownload"),
            Err(
                "field `expected_failure` expects `SetupFailure(kind, \"reason\")` or \
                 `TestFailure(\"reason\")`"
                    .to_owned()
            )
        );
    }

    #[test]
    fn an_empty_reason_is_refused() {
        assert_eq!(
            [
                parse("SetupFailure(ModelDownload, \"\")"),
                parse("TestFailure(\"\")")
            ],
            [(); 2].map(|()| Err(
                "`expected_failure` expects a non-empty string stating why the trial fails on \
                 purpose"
                    .to_owned()
            ))
        );
    }

    #[test]
    fn a_setup_failure_kind_that_is_not_a_bare_name_is_refused() {
        assert_eq!(
            parse("SetupFailure(kinds::ModelDownload, \"reason\")"),
            Err(
                "`SetupFailure` expects the bare name of a phase setup failure kind, such as \
                 `ModelDownload`"
                    .to_owned()
            )
        );
    }
}
