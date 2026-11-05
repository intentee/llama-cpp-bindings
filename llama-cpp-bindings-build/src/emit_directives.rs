use std::io::Write;

use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;

/// Writes `directives` for cargo, one per line, leaving the debug messages out unless
/// `include_debug` asks for them.
///
/// # Errors
///
/// Returns [`BuildError::DirectiveWrite`] when `output` refuses a line.
pub fn emit_directives(
    directives: &[CargoDirective],
    include_debug: bool,
    output: &mut impl Write,
) -> Result<(), BuildError> {
    directives
        .iter()
        .filter(|directive| include_debug || !directive.is_debug())
        .try_for_each(|directive| writeln!(output, "{directive}"))
        .map_err(BuildError::DirectiveWrite)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::PathBuf;

    use super::emit_directives;
    use crate::cargo_directive::CargoDirective;

    fn rebuild_and_debug_directives() -> [CargoDirective; 2] {
        [
            CargoDirective::RerunIfChanged(PathBuf::from("wrapper.h")),
            CargoDirective::Debug("Bindings Created".to_owned()),
        ]
    }

    fn emitted(include_debug: bool) -> String {
        let mut output = Vec::new();

        emit_directives(&rebuild_and_debug_directives(), include_debug, &mut output).unwrap();

        String::from_utf8(output).unwrap()
    }

    #[test]
    fn debug_messages_are_left_out_unless_asked_for() {
        assert_eq!(
            [emitted(false), emitted(true)],
            [
                "cargo:rerun-if-changed=wrapper.h\n".to_owned(),
                "cargo:rerun-if-changed=wrapper.h\ncargo:warning=[DEBUG] Bindings Created\n"
                    .to_owned()
            ]
        );
    }

    #[test]
    fn an_output_that_refuses_writes_is_reported() {
        let mut read_only = File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")).unwrap();

        assert_eq!(
            emit_directives(&rebuild_and_debug_directives(), false, &mut read_only)
                .err()
                .map(|error| error.to_string()),
            Some(
                "cargo directives could not be written: Bad file descriptor (os error 9)"
                    .to_owned()
            )
        );
    }
}
