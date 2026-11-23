use crate::frame_stack::FrameStack;
use crate::recorded_error::RecordedError;

/// An RAII capture scope for errors raised inside FFI callbacks.
///
/// While an `ErrorScope` is alive, an error recorded via [`crate::record`] on
/// the same thread is captured in this scope's frame. Scopes nest: each
/// [`ErrorScope::enter`] pushes its own frame and [`Drop`] pops it, so an inner
/// FFI call cannot leak an error into an outer one.
#[derive(Debug)]
#[non_exhaustive]
pub struct ErrorScope {
    frame_index: usize,
}

impl ErrorScope {
    #[must_use]
    pub fn enter() -> Self {
        Self {
            frame_index: FrameStack::push_frame(),
        }
    }

    #[must_use]
    pub fn take(&self) -> Option<RecordedError> {
        FrameStack::take_from_frame(self.frame_index)
    }
}

impl Drop for ErrorScope {
    fn drop(&mut self) {
        FrameStack::pop_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorScope;
    use crate::record::record;
    use crate::recorded_error::RecordedError;

    #[test]
    fn take_without_a_recorded_error_is_none() {
        let scope = ErrorScope::enter();

        assert!(scope.take().is_none());
    }

    #[test]
    fn take_consumes_the_recorded_error() {
        let scope = ErrorScope::enter();
        record("once".to_string());

        assert_eq!(
            scope.take().map(RecordedError::into_message),
            Some("once".to_string())
        );
        assert!(scope.take().is_none());
    }

    #[test]
    fn nested_scopes_capture_independently() {
        let outer = ErrorScope::enter();
        {
            let inner = ErrorScope::enter();
            record("inner".to_string());

            assert_eq!(
                inner.take().map(RecordedError::into_message),
                Some("inner".to_string())
            );
        }

        record("outer".to_string());

        assert_eq!(
            outer.take().map(RecordedError::into_message),
            Some("outer".to_string())
        );
    }

    #[test]
    fn an_outer_scope_takes_its_own_error_while_an_inner_scope_is_alive() {
        let outer = ErrorScope::enter();
        record("outer".to_string());
        let inner = ErrorScope::enter();
        record("inner".to_string());

        assert_eq!(
            [&outer, &inner].map(|scope| scope.take().map(RecordedError::into_message)),
            [Some("outer".to_string()), Some("inner".to_string())]
        );
    }
}
