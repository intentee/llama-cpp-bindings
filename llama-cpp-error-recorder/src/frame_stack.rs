use std::cell::RefCell;
use std::error::Error;

use log::error;

use crate::recorded_error::RecordedError;

thread_local! {
    static FRAMES: RefCell<Vec<Option<RecordedError>>> = const { RefCell::new(Vec::new()) };
}

pub struct FrameStack;

impl FrameStack {
    pub fn pop_frame() {
        FRAMES.with(|cell| {
            cell.borrow_mut().pop();
        });
    }

    pub fn push_frame() -> usize {
        FRAMES.with(|cell| {
            let mut frames = cell.borrow_mut();
            let frame_index = frames.len();
            frames.push(None);

            frame_index
        })
    }

    pub fn record_into_top(error: impl Into<Box<dyn Error + Send + Sync>>) {
        let error = RecordedError::new(error);

        FRAMES.with(|cell| {
            let mut frames = cell.borrow_mut();
            let Some(top) = frames.last_mut() else {
                error!("an FFI callback failed outside any error scope: {error}");

                return;
            };

            if top.is_some() {
                error!("an FFI callback failed again after the root cause was recorded: {error}");

                return;
            }

            *top = Some(error);
        });
    }

    pub fn take_from_frame(frame_index: usize) -> Option<RecordedError> {
        FRAMES.with(|cell| {
            cell.borrow_mut()
                .get_mut(frame_index)
                .and_then(Option::take)
        })
    }
}

#[cfg(test)]
mod tests {
    use log::Level;

    use llama_cpp_capturing_logger::captured_record::CapturedRecord;
    use llama_cpp_capturing_logger::capturing_logger::CapturingLogger;

    use super::FrameStack;
    use crate::recorded_error::RecordedError;

    const LOG_TARGET: &str = "llama_cpp_error_recorder::frame_stack";

    fn logged(message: &str) -> bool {
        CapturingLogger::records_for(LOG_TARGET).contains(&CapturedRecord {
            level: Level::Error,
            target: LOG_TARGET.to_owned(),
            message: message.to_owned(),
        })
    }

    #[test]
    fn an_error_recorded_outside_any_frame_is_logged_and_reaches_no_later_frame() {
        CapturingLogger::install();

        FrameStack::record_into_top("an orphaned failure".to_string());

        let frame_index = FrameStack::push_frame();
        let taken = FrameStack::take_from_frame(frame_index);

        FrameStack::pop_frame();

        assert!(taken.is_none());
        assert!(logged(
            "an FFI callback failed outside any error scope: an orphaned failure"
        ));
    }

    #[test]
    fn a_frame_keeps_its_first_error_and_logs_every_later_one() {
        CapturingLogger::install();

        let frame_index = FrameStack::push_frame();

        FrameStack::record_into_top("the root cause".to_string());
        FrameStack::record_into_top("a later failure".to_string());

        let taken = FrameStack::take_from_frame(frame_index);

        FrameStack::pop_frame();

        assert_eq!(
            taken.map(RecordedError::into_message),
            Some("the root cause".to_string())
        );
        assert!(logged(
            "an FFI callback failed again after the root cause was recorded: a later failure"
        ));
    }
}
