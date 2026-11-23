use std::sync::Mutex;
use std::sync::Once;
use std::sync::PoisonError;

use log::LevelFilter;
use log::Log;
use log::Metadata;
use log::Record;

use crate::captured_record::CapturedRecord;

static CAPTURING_LOGGER: CapturingLogger = CapturingLogger {
    records: Mutex::new(Vec::new()),
};

static INSTALL: Once = Once::new();

pub struct CapturingLogger {
    records: Mutex<Vec<CapturedRecord>>,
}

impl CapturingLogger {
    /// Makes the capturing logger the logger of this process, once.
    ///
    /// # Panics
    ///
    /// Panics when another logger was installed first, because the records the tests assert on
    /// would never reach this one.
    pub fn install() {
        INSTALL.call_once(|| {
            assert!(
                log::set_logger(&CAPTURING_LOGGER).is_ok(),
                "the capturing logger must be the only logger of the test process"
            );
            log::set_max_level(LevelFilter::Trace);
        });
    }

    #[must_use]
    pub fn records_for(target: &str) -> Vec<CapturedRecord> {
        CAPTURING_LOGGER
            .records
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(
                |CapturedRecord {
                     target: record_target,
                     ..
                 }| record_target == target,
            )
            .cloned()
            .collect()
    }
}

impl Log for CapturingLogger {
    fn enabled(&self, _: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        self.records
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(CapturedRecord {
                level: record.level(),
                target: record.target().to_owned(),
                message: record.args().to_string(),
            });
    }

    fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    use log::Level;
    use log::Log;
    use log::Metadata;

    use super::CAPTURING_LOGGER;

    #[test]
    fn the_capturing_logger_enables_every_record() {
        let metadata = Metadata::builder()
            .level(Level::Info)
            .target("test-logger-enabled")
            .build();

        assert!(CAPTURING_LOGGER.enabled(&metadata));
        CAPTURING_LOGGER.flush();
    }
}
