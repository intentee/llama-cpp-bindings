use std::path::Path;

use syn::Expr;
use syn::ExprArray;
use syn::ExprCall;
use syn::ExprLit;
use syn::ExprPath;
use syn::Lit;

fn require_string_argument(expression: &Expr, field: &str, position: &str) -> syn::Result<String> {
    match expression {
        Expr::Lit(ExprLit {
            lit: Lit::Str(string_literal),
            ..
        }) => Ok(string_literal.value()),
        _ => Err(syn::Error::new_spanned(
            expression,
            format!("`{field}` argument `{position}` expects a string literal"),
        )),
    }
}

fn parse_huggingface_source(
    ExprCall {
        args: call_arguments,
        ..
    }: &ExprCall,
    field: &str,
) -> syn::Result<ParsedSource> {
    let args: Vec<&Expr> = call_arguments.iter().collect();
    let [repo_expr, file_expr] = args.as_slice() else {
        return Err(syn::Error::new_spanned(
            call_arguments,
            format!(
                "`HuggingFace` expects exactly 2 string arguments (repo, file); got {got}",
                got = call_arguments.len()
            ),
        ));
    };
    let repo = require_string_argument(repo_expr, field, "HuggingFace.repo")?;
    let file = require_string_argument(file_expr, field, "HuggingFace.file")?;
    Ok(ParsedSource::HuggingFace { repo, file })
}

fn parse_huggingface_shards_source(
    ExprCall {
        args: call_arguments,
        ..
    }: &ExprCall,
    field: &str,
) -> syn::Result<ParsedSource> {
    let args: Vec<&Expr> = call_arguments.iter().collect();
    let [repo_expr, commit_expr, shards_expr] = args.as_slice() else {
        return Err(syn::Error::new_spanned(
            call_arguments,
            format!(
                "`HuggingFaceShards` expects exactly 3 arguments (repo, commit, [shards]); got {got}",
                got = call_arguments.len()
            ),
        ));
    };
    let repo = require_string_argument(repo_expr, field, "HuggingFaceShards.repo")?;
    let commit = require_string_argument(commit_expr, field, "HuggingFaceShards.commit")?;
    let Expr::Array(ExprArray {
        elems: shard_exprs, ..
    }) = shards_expr
    else {
        return Err(syn::Error::new_spanned(
            shards_expr,
            format!(
                "`{field}` argument `HuggingFaceShards.shards` expects an array of string literals"
            ),
        ));
    };
    let shards = shard_exprs
        .iter()
        .map(|shard_expr| require_string_argument(shard_expr, field, "HuggingFaceShards.shards"))
        .collect::<syn::Result<Vec<String>>>()?;

    match shards.as_slice() {
        [first_shard, remaining_shards @ ..] if !remaining_shards.is_empty() => {
            Ok(ParsedSource::HuggingFaceShards {
                repo,
                commit,
                first_shard: first_shard.clone(),
                remaining_shards: remaining_shards.to_vec(),
            })
        }
        _ => Err(syn::Error::new_spanned(
            shards_expr,
            "`HuggingFaceShards` expects at least 2 shards; a single-file model is `HuggingFace(repo, file)`",
        )),
    }
}

fn parse_local_path_source(
    ExprCall {
        args: call_arguments,
        ..
    }: &ExprCall,
    field: &str,
) -> syn::Result<ParsedSource> {
    let args: Vec<&Expr> = call_arguments.iter().collect();
    let [path_expr] = args.as_slice() else {
        return Err(syn::Error::new_spanned(
            call_arguments,
            format!(
                "`LocalPath` expects exactly 1 string argument (path); got {got}",
                got = call_arguments.len()
            ),
        ));
    };
    let path = require_string_argument(path_expr, field, "LocalPath.path")?;
    Ok(ParsedSource::LocalPath(path))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParsedSource {
    HuggingFace {
        repo: String,
        file: String,
    },
    HuggingFaceShards {
        repo: String,
        commit: String,
        first_shard: String,
        remaining_shards: Vec<String>,
    },
    LocalPath(String),
}

impl ParsedSource {
    pub fn parse(expression: &Expr, field: &str) -> syn::Result<Self> {
        let Expr::Call(call) = expression else {
            return Err(syn::Error::new_spanned(
                expression,
                format!(
                    "field `{field}` expects `HuggingFace(repo, file)`, `HuggingFaceShards(repo, commit, [shards])` or `LocalPath(path)`"
                ),
            ));
        };
        let Expr::Path(ExprPath { path, .. }) = call.func.as_ref() else {
            return Err(syn::Error::new_spanned(
                call.func.as_ref(),
                format!(
                    "field `{field}` expects `HuggingFace(...)`, `HuggingFaceShards(...)` or `LocalPath(...)`"
                ),
            ));
        };
        let variant_ident = path.get_ident().ok_or_else(|| {
            syn::Error::new_spanned(
                path,
                format!(
                    "field `{field}` expects the bare variant name `HuggingFace`, `HuggingFaceShards` or `LocalPath`"
                ),
            )
        })?;
        match variant_ident.to_string().as_str() {
            "HuggingFace" => parse_huggingface_source(call, field),
            "HuggingFaceShards" => parse_huggingface_shards_source(call, field),
            "LocalPath" => parse_local_path_source(call, field),
            other => Err(syn::Error::new_spanned(
                variant_ident,
                format!(
                    "unknown source variant `{other}`; expected `HuggingFace`, `HuggingFaceShards` or `LocalPath`"
                ),
            )),
        }
    }

    pub fn display_suffix(&self) -> String {
        match self {
            Self::HuggingFace { file, .. } => file.clone(),
            Self::HuggingFaceShards { first_shard, .. } => first_shard.clone(),
            Self::LocalPath(path) => Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .map_or_else(|| path.clone(), str::to_owned),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use syn::parse_str;

    use super::ParsedSource;

    fn parse(source: &str) -> syn::Result<ParsedSource> {
        let expression: syn::Expr =
            parse_str(source).expect("the test source must be an expression");
        ParsedSource::parse(&expression, "model_source")
    }

    /// A model split into three shards, as `HuggingFaceShards` declares it.
    #[must_use]
    pub fn three_shard_source() -> ParsedSource {
        ParsedSource::HuggingFaceShards {
            repo: "org/name".to_owned(),
            commit: "abc123".to_owned(),
            first_shard: "m-00001-of-00003.gguf".to_owned(),
            remaining_shards: vec![
                "m-00002-of-00003.gguf".to_owned(),
                "m-00003-of-00003.gguf".to_owned(),
            ],
        }
    }

    #[test]
    fn parses_huggingface_with_two_string_args() {
        let parsed = parse("HuggingFace(\"org/name\", \"file.gguf\")").expect("valid");

        assert_eq!(
            parsed,
            ParsedSource::HuggingFace {
                repo: "org/name".to_owned(),
                file: "file.gguf".to_owned(),
            },
        );
    }

    #[test]
    fn parses_huggingface_shards_into_the_first_shard_and_the_rest() {
        let parsed = parse("HuggingFaceShards(\"org/name\", \"abc123\", [\"m-00001-of-00003.gguf\", \"m-00002-of-00003.gguf\", \"m-00003-of-00003.gguf\"])")
            .expect("valid");

        assert_eq!(parsed, three_shard_source());
    }

    #[test]
    fn huggingface_shards_with_a_single_shard_is_rejected() {
        let message = parse("HuggingFaceShards(\"org/name\", \"abc123\", [\"m.gguf\"])")
            .expect_err("a single shard must fail")
            .to_string();

        assert_eq!(
            message,
            "`HuggingFaceShards` expects at least 2 shards; a single-file model is `HuggingFace(repo, file)`"
        );
    }

    #[test]
    fn huggingface_shards_without_an_array_of_shards_is_rejected() {
        let message = parse("HuggingFaceShards(\"org/name\", \"abc123\", \"m.gguf\")")
            .expect_err("a non-array shards argument must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `HuggingFaceShards.shards` expects an array of string literals"
        );
    }

    #[test]
    fn huggingface_shards_with_a_non_string_shard_is_rejected() {
        let message = parse("HuggingFaceShards(\"org/name\", \"abc123\", [\"m.gguf\", 42])")
            .expect_err("a non-string shard must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `HuggingFaceShards.shards` expects a string literal"
        );
    }

    #[test]
    fn huggingface_shards_with_a_non_string_repo_is_rejected() {
        let message = parse("HuggingFaceShards(42, \"abc123\", [\"a.gguf\", \"b.gguf\"])")
            .expect_err("a non-string repo must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `HuggingFaceShards.repo` expects a string literal"
        );
    }

    #[test]
    fn huggingface_shards_with_a_non_string_commit_is_rejected() {
        let message = parse("HuggingFaceShards(\"org/name\", 42, [\"a.gguf\", \"b.gguf\"])")
            .expect_err("a non-string commit must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `HuggingFaceShards.commit` expects a string literal"
        );
    }

    #[test]
    fn huggingface_shards_with_wrong_arity_is_rejected() {
        let message = parse("HuggingFaceShards(\"org/name\", [\"a.gguf\", \"b.gguf\"])")
            .expect_err("arity must fail")
            .to_string();

        assert_eq!(
            message,
            "`HuggingFaceShards` expects exactly 3 arguments (repo, commit, [shards]); got 2"
        );
    }

    #[test]
    fn parses_local_path_with_one_string_arg() {
        let parsed = parse("LocalPath(\"/abs/local.gguf\")").expect("valid");

        assert_eq!(
            parsed,
            ParsedSource::LocalPath("/abs/local.gguf".to_owned())
        );
    }

    #[test]
    fn unknown_variant_is_rejected() {
        let message = parse("Mystery(\"a\", \"b\")")
            .expect_err("unknown variant must fail")
            .to_string();

        assert_eq!(
            message,
            "unknown source variant `Mystery`; expected `HuggingFace`, `HuggingFaceShards` or `LocalPath`"
        );
    }

    #[test]
    fn non_call_expression_is_rejected() {
        let message = parse("\"plain\"")
            .expect_err("non-call must fail")
            .to_string();

        assert_eq!(
            message,
            "field `model_source` expects `HuggingFace(repo, file)`, `HuggingFaceShards(repo, commit, [shards])` or `LocalPath(path)`"
        );
    }

    #[test]
    fn huggingface_with_wrong_arity_is_rejected() {
        let message = parse("HuggingFace(\"only-one\")")
            .expect_err("arity must fail")
            .to_string();

        assert_eq!(
            message,
            "`HuggingFace` expects exactly 2 string arguments (repo, file); got 1"
        );
    }

    #[test]
    fn local_path_with_wrong_arity_is_rejected() {
        let message = parse("LocalPath(\"a\", \"b\")")
            .expect_err("arity must fail")
            .to_string();

        assert_eq!(
            message,
            "`LocalPath` expects exactly 1 string argument (path); got 2"
        );
    }

    #[test]
    fn huggingface_with_non_string_first_argument_is_rejected() {
        let message = parse("HuggingFace(42, \"file\")")
            .expect_err("non-string arg must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `HuggingFace.repo` expects a string literal"
        );
    }

    #[test]
    fn huggingface_with_non_string_second_argument_is_rejected() {
        let message = parse("HuggingFace(\"repo\", 42)")
            .expect_err("non-string second arg must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `HuggingFace.file` expects a string literal"
        );
    }

    #[test]
    fn local_path_with_non_string_argument_is_rejected() {
        let message = parse("LocalPath(42)")
            .expect_err("non-string LocalPath arg must fail")
            .to_string();

        assert_eq!(
            message,
            "`model_source` argument `LocalPath.path` expects a string literal"
        );
    }

    #[test]
    fn non_path_function_expression_is_rejected() {
        let message = parse("(closure)(\"a\")")
            .expect_err("non-path func must fail")
            .to_string();

        assert_eq!(
            message,
            "field `model_source` expects `HuggingFace(...)`, `HuggingFaceShards(...)` or `LocalPath(...)`"
        );
    }

    #[test]
    fn qualified_path_variant_is_rejected() {
        let message = parse("some::Other::HuggingFace(\"a\", \"b\")")
            .expect_err("qualified path must fail")
            .to_string();

        assert_eq!(
            message,
            "field `model_source` expects the bare variant name `HuggingFace`, `HuggingFaceShards` or `LocalPath`"
        );
    }

    #[test]
    fn display_suffix_huggingface_returns_file() {
        let source = ParsedSource::HuggingFace {
            repo: "org/name".to_owned(),
            file: "model.gguf".to_owned(),
        };

        assert_eq!(source.display_suffix(), "model.gguf");
    }

    #[test]
    fn display_suffix_huggingface_shards_returns_the_first_shard() {
        let source = ParsedSource::HuggingFaceShards {
            repo: "org/name".to_owned(),
            commit: "abc123".to_owned(),
            first_shard: "m-00001-of-00002.gguf".to_owned(),
            remaining_shards: vec!["m-00002-of-00002.gguf".to_owned()],
        };

        assert_eq!(source.display_suffix(), "m-00001-of-00002.gguf");
    }

    #[test]
    fn display_suffix_local_path_returns_basename() {
        let source = ParsedSource::LocalPath("/abs/dir/model.gguf".to_owned());

        assert_eq!(source.display_suffix(), "model.gguf");
    }

    #[test]
    fn display_suffix_local_path_without_file_name_returns_full_path() {
        let source = ParsedSource::LocalPath("/abs/dir/..".to_owned());

        assert_eq!(source.display_suffix(), "/abs/dir/..");
    }
}
