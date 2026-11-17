use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_bindings_types::token_usage::TokenUsage;

use crate::classify_sample_loop_outcome::ClassifySampleLoopOutcome;

/// A generated answer to a chat session, with what the classifier attributed to each section,
/// the session's thinking tags, and the classifier's own final parse of the answer.
#[derive(Debug)]
pub struct ChatAnswer {
    pub outcome: ClassifySampleLoopOutcome,
    pub usage: TokenUsage,
    pub thinking_tags: Vec<String>,
    pub parse: ChatMessageParseOutcome,
}

impl ChatAnswer {
    /// Asserts that every generated token is content, and that no thinking tag of the session
    /// reached the content stream.
    ///
    /// # Panics
    ///
    /// Panics when the answer reasons, holds a token of no section, or leaks a thinking tag.
    pub fn assert_content_without_reasoning(&self) {
        let Self {
            outcome,
            usage,
            thinking_tags,
            ..
        } = self;

        assert_ne!(outcome.generated.text(), "", "{outcome:?}");
        assert!(outcome.observed_content > 0, "{outcome:?}");
        assert_eq!(outcome.observed_reasoning, 0, "{outcome:?}");
        assert_eq!(outcome.observed_undeterminable, 0, "{outcome:?}");
        assert_eq!(usage.reasoning, 0, "{usage:?}");
        assert_eq!(usage.undeterminable, 0, "{usage:?}");
        assert_eq!(
            usage.completion_tokens(),
            outcome.observed_content,
            "{usage:?}"
        );

        for tag in thinking_tags {
            assert!(
                !outcome.content_stream.contains(tag),
                "the content stream leaked the thinking tag {tag:?}; {outcome:?}"
            );
        }
    }

    /// Asserts that the answer reasons, and that the reasoning and content streams are exactly
    /// what the classifier's own final parse extracts, apart from `prompted_reasoning`, the reasoning text
    /// the generation prompt itself holds, which the parser shows in front of the first
    /// generated text.
    ///
    /// # Errors
    ///
    /// Fails when the parser fails on or does not recognize the answer, or when the answer
    /// does not reason, holds a token of no section, leaks a thinking tag, or streams text the
    /// parser does not extract.
    pub fn assert_reasoning_matches_parse(&self, prompted_reasoning: &str) -> Result<()> {
        let Self {
            outcome,
            usage,
            thinking_tags,
            parse,
        } = self;
        ensure!(
            matches!(parse, ChatMessageParseOutcome::Recognized(_)),
            "the session's parser must recognize the generated answer: {parse:?}"
        );

        let parsed = parse
            .clone()
            .into_recognized()
            .ok()
            .context("the session's parser must recognize the generated answer")?;
        let observed_completion = outcome
            .observed_content
            .checked_add(outcome.observed_reasoning)
            .context("the observed completion token count must fit u64")?;

        ensure!(outcome.observed_reasoning > 0, "{outcome:?}");
        ensure!(outcome.observed_undeterminable == 0, "{outcome:?}");
        ensure!(usage.reasoning == outcome.observed_reasoning, "{usage:?}");
        ensure!(usage.undeterminable == 0, "{usage:?}");
        ensure!(
            usage.completion_tokens() == observed_completion,
            "{usage:?}"
        );
        ensure!(!parsed.reasoning_content.is_empty(), "{outcome:?}");
        ensure!(
            format!("{prompted_reasoning}{}", outcome.reasoning_stream) == parsed.reasoning_content
        );
        ensure!(outcome.content_stream == parsed.content);

        for tag in thinking_tags {
            ensure!(!outcome.reasoning_stream.contains(tag), "{outcome:?}");
            ensure!(!outcome.content_stream.contains(tag), "{outcome:?}");
        }

        Ok(())
    }
}
