use crate::decode_anomaly::DecodeAnomaly;
use crate::decode_output::DecodeOutput;
use crate::decode_result::DecodeResult;
use crate::incoming_log_level::IncomingLogLevel;
use crate::log_level::LogLevel;
use crate::log_line::LogLine;

pub struct LogDecoder {
    buffered: Option<LogLine>,
    previous_level: LogLevel,
}

impl LogDecoder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffered: None,
            previous_level: LogLevel::None,
        }
    }

    pub fn feed(&mut self, level: IncomingLogLevel, text: &str) -> DecodeResult {
        match level {
            IncomingLogLevel::Cont => self.feed_cont(text),
            IncomingLogLevel::Debug => self.feed_non_cont(LogLevel::Debug, text),
            IncomingLogLevel::Error => self.feed_non_cont(LogLevel::Error, text),
            IncomingLogLevel::Info => self.feed_non_cont(LogLevel::Info, text),
            IncomingLogLevel::None => self.feed_non_cont(LogLevel::None, text),
            IncomingLogLevel::Unknown(raw) => self.feed_non_cont(LogLevel::Unknown(raw), text),
            IncomingLogLevel::Warn => self.feed_non_cont(LogLevel::Warn, text),
        }
    }

    fn feed_cont(&mut self, text: &str) -> DecodeResult {
        if let Some(LogLine {
            level,
            text: mut buffer,
        }) = self.buffered.take()
        {
            buffer.push_str(text);
            if let Some(without_newline) = buffer.strip_suffix('\n') {
                DecodeResult {
                    output: DecodeOutput::Line(LogLine {
                        level,
                        text: without_newline.to_owned(),
                    }),
                    anomaly: None,
                }
            } else {
                self.buffered = Some(LogLine {
                    level,
                    text: buffer,
                });
                DecodeResult {
                    output: DecodeOutput::None,
                    anomaly: None,
                }
            }
        } else {
            self.feed_orphan_cont(text)
        }
    }

    fn feed_orphan_cont(&mut self, text: &str) -> DecodeResult {
        let level = self.previous_level;
        if let Some(without_newline) = text.strip_suffix('\n') {
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level,
                    text: without_newline.to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        } else {
            self.buffered = Some(LogLine {
                level,
                text: text.to_owned(),
            });
            DecodeResult {
                output: DecodeOutput::None,
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        }
    }

    fn feed_non_cont(&mut self, level: LogLevel, text: &str) -> DecodeResult {
        self.previous_level = level;
        let stale = self.buffered.take();
        match (text.strip_suffix('\n'), stale) {
            (
                Some(without_newline),
                Some(LogLine {
                    level: stale_level,
                    text: stale_text,
                }),
            ) => DecodeResult {
                output: DecodeOutput::TwoLines {
                    earlier: LogLine {
                        level: stale_level,
                        text: stale_text,
                    },
                    current: LogLine {
                        level,
                        text: without_newline.to_owned(),
                    },
                },
                anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
            },
            (Some(without_newline), None) => DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level,
                    text: without_newline.to_owned(),
                }),
                anomaly: None,
            },
            (
                None,
                Some(LogLine {
                    level: stale_level,
                    text: stale_text,
                }),
            ) => {
                self.buffered = Some(LogLine {
                    level,
                    text: text.to_owned(),
                });
                DecodeResult {
                    output: DecodeOutput::Line(LogLine {
                        level: stale_level,
                        text: stale_text,
                    }),
                    anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
                }
            }
            (None, None) => {
                self.buffered = Some(LogLine {
                    level,
                    text: text.to_owned(),
                });
                DecodeResult {
                    output: DecodeOutput::None,
                    anomaly: None,
                }
            }
        }
    }
}

impl Default for LogDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::LogDecoder;
    use crate::decode_anomaly::DecodeAnomaly;
    use crate::decode_output::DecodeOutput;
    use crate::decode_result::DecodeResult;
    use crate::incoming_log_level::IncomingLogLevel;
    use crate::log_level::LogLevel;
    use crate::log_line::LogLine;

    #[test]
    fn a_complete_line_is_decoded_without_its_newline() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Info, "hello\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "hello".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn a_line_without_a_newline_is_held_back() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Info, "hello");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::None,
                anomaly: None,
            }
        );
    }

    #[test]
    fn a_continuation_completes_the_held_line() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "hello ");
        let result = decoder.feed(IncomingLogLevel::Cont, "world\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "hello world".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn continuations_join_until_the_newline() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "part1 ");
        decoder.feed(IncomingLogLevel::Cont, "part2 ");
        let result = decoder.feed(IncomingLogLevel::Cont, "part3\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "part1 part2 part3".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn a_new_complete_line_flushes_the_held_line_as_stale() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "stale");
        let result = decoder.feed(IncomingLogLevel::Warn, "fresh\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::TwoLines {
                    earlier: LogLine {
                        level: LogLevel::Info,
                        text: "stale".to_owned(),
                    },
                    current: LogLine {
                        level: LogLevel::Warn,
                        text: "fresh".to_owned(),
                    },
                },
                anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
            }
        );
    }

    #[test]
    fn a_new_partial_line_flushes_the_held_line_and_is_held_in_its_place() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Info, "first");
        let result = decoder.feed(IncomingLogLevel::Warn, "second");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Info,
                    text: "first".to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::StaleBufferAbandoned),
            }
        );

        let follow_up = decoder.feed(IncomingLogLevel::Cont, "more\n");
        assert_eq!(
            follow_up,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Warn,
                    text: "secondmore".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn a_continuation_without_any_earlier_line_is_an_orphan_anomaly() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Cont, "ghost\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::None,
                    text: "ghost".to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        );
    }

    #[test]
    fn an_orphan_continuation_takes_the_previous_line_level() {
        let mut decoder = LogDecoder::new();
        decoder.feed(IncomingLogLevel::Warn, "complete\n");
        let result = decoder.feed(IncomingLogLevel::Cont, "ghost\n");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::Warn,
                    text: "ghost".to_owned(),
                }),
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        );
    }

    #[test]
    fn a_complete_line_keeps_the_level_it_arrived_with() {
        let incoming_levels = [
            IncomingLogLevel::None,
            IncomingLogLevel::Unknown(9999),
            IncomingLogLevel::Debug,
            IncomingLogLevel::Error,
        ];

        assert_eq!(
            incoming_levels.map(|incoming_level| LogDecoder::new().feed(incoming_level, "line\n")),
            [
                LogLevel::None,
                LogLevel::Unknown(9999),
                LogLevel::Debug,
                LogLevel::Error
            ]
            .map(|level| {
                DecodeResult {
                    output: DecodeOutput::Line(LogLine {
                        level,
                        text: "line".to_owned(),
                    }),
                    anomaly: None,
                }
            })
        );
    }

    #[test]
    fn feed_orphan_cont_without_newline_buffers_text() {
        let mut decoder = LogDecoder::new();
        let result = decoder.feed(IncomingLogLevel::Cont, "fragment");

        assert_eq!(
            result,
            DecodeResult {
                output: DecodeOutput::None,
                anomaly: Some(DecodeAnomaly::OrphanCont),
            }
        );

        let follow_up = decoder.feed(IncomingLogLevel::Cont, " rest\n");
        assert_eq!(
            follow_up,
            DecodeResult {
                output: DecodeOutput::Line(LogLine {
                    level: LogLevel::None,
                    text: "fragment rest".to_owned(),
                }),
                anomaly: None,
            }
        );
    }

    #[test]
    fn a_default_decoder_decodes_like_a_new_one() {
        let mut default_decoder = LogDecoder::default();
        let new_decoder_result = LogDecoder::new().feed(IncomingLogLevel::Info, "compare\n");
        let default_result = default_decoder.feed(IncomingLogLevel::Info, "compare\n");

        assert_eq!(default_result, new_decoder_result);
    }
}
