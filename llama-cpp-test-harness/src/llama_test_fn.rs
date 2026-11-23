use crate::llama_fixture::LlamaFixture;

#[derive(Clone, Copy)]
pub enum LlamaTestFn {
    Fallible(
        for<'reference, 'fixture> fn(&'reference LlamaFixture<'fixture>) -> anyhow::Result<()>,
    ),
    Infallible(for<'reference, 'fixture> fn(&'reference LlamaFixture<'fixture>)),
}

impl LlamaTestFn {
    /// A test with an empty body, which passes whenever its phase sets up.
    #[must_use]
    pub const fn no_op() -> Self {
        Self::Infallible(|_fixture| {})
    }

    /// Runs the test against `fixture`; a test that cannot fail by returning an error succeeds
    /// whenever it returns.
    ///
    /// # Errors
    ///
    /// Forwards the error a fallible test returns.
    pub fn run(self, fixture: &LlamaFixture<'_>) -> anyhow::Result<()> {
        match self {
            Self::Fallible(test) => test(fixture),
            Self::Infallible(test) => {
                test(fixture);

                Ok(())
            }
        }
    }
}
