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
