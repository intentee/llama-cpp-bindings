use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ChatOutputCompleteness;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::TokenUsage;

use crate::classify_sample_loop_outcome::ClassifySampleLoopOutcome;
use crate::recognized_answer::recognized_answer;

const REASONING_PROBE: &str = "x";

fn thinking_tags(session: &ChatSession<'_>) -> Result<Vec<String>> {
    let mut tags = session.thinking_end_tags()?;

    tags.push(session.thinking_start_tag()?);

    Ok(tags)
}

fn prompted_reasoning(session: &ChatSession<'_>) -> Result<String> {
    let probed = recognized_answer(session, REASONING_PROBE, ChatOutputCompleteness::Partial)?;

    if probed.content == REASONING_PROBE && probed.reasoning_content.is_empty() {
        return Ok(String::new());
    }

    let Some(prompted) = probed.reasoning_content.strip_suffix(REASONING_PROBE) else {
        bail!("the probe must continue either the prompt's reasoning or its content; {probed:?}");
    };

    Ok(prompted.to_owned())
}

/// A generated answer to a chat session, with what the classifier attributed to each section.
#[derive(Debug)]
pub struct ChatAnswer {
    pub outcome: ClassifySampleLoopOutcome,
    pub usage: TokenUsage,
}

impl ChatAnswer {
    /// Asserts that every generated token is content, and that no thinking tag of `session`
    /// reached the content stream.
    ///
    /// # Errors
    ///
    /// Forwards errors of reading the session's thinking tags.
    ///
    /// # Panics
    ///
    /// Panics when the answer reasons, holds a token of no section, or leaks a thinking tag.
    pub fn assert_content_without_reasoning(&self, session: &ChatSession<'_>) -> Result<()> {
        let Self { outcome, usage } = self;

        assert_ne!(outcome.generated.text(), "", "{outcome:?}");
        assert!(outcome.observed_content > 0, "{outcome:?}");
        assert_eq!(outcome.observed_reasoning, 0, "{outcome:?}");
        assert_eq!(outcome.observed_undeterminable, 0, "{outcome:?}");
        assert_eq!(usage.reasoning_tokens, 0, "{usage:?}");
        assert_eq!(usage.undeterminable_tokens, 0, "{usage:?}");
        assert_eq!(
            usage.completion_tokens(),
            outcome.observed_content,
            "{usage:?}"
        );

        for tag in thinking_tags(session)? {
            assert!(
                !outcome.content_stream.contains(&tag),
                "the content stream leaked the thinking tag {tag:?}; {outcome:?}"
            );
        }

        Ok(())
    }

    /// Asserts that the answer reasons, and that the reasoning and content streams are exactly
    /// what the session's parser extracts, apart from the reasoning text the generation prompt
    /// itself holds, which the parser shows in front of the first generated text.
    ///
    /// # Errors
    ///
    /// Forwards errors of parsing the answer or reading the session's thinking tags, and fails
    /// when the parser does not recognize the answer.
    ///
    /// # Panics
    ///
    /// Panics when the answer does not reason, holds a token of no section, leaks a thinking
    /// tag, or streams text the parser does not extract.
    pub fn assert_reasoning_matches_parse(&self, session: &ChatSession<'_>) -> Result<()> {
        let Self { outcome, usage } = self;
        let ChatMessageParseOutcome::Recognized(parsed) =
            session.parse(&outcome.generated, ChatOutputCompleteness::Finished)?
        else {
            bail!("the session's parser must recognize the generated answer; {outcome:?}");
        };
        let prompted_reasoning = prompted_reasoning(session)?;

        assert!(outcome.observed_reasoning > 0, "{outcome:?}");
        assert_eq!(outcome.observed_undeterminable, 0, "{outcome:?}");
        assert_eq!(
            usage.reasoning_tokens, outcome.observed_reasoning,
            "{usage:?}"
        );
        assert_eq!(usage.undeterminable_tokens, 0, "{usage:?}");
        assert_eq!(
            usage.completion_tokens(),
            outcome.observed_content + outcome.observed_reasoning,
            "{usage:?}"
        );
        assert_ne!(parsed.reasoning_content, "", "{outcome:?}");
        assert_eq!(
            format!("{prompted_reasoning}{}", outcome.reasoning_stream),
            parsed.reasoning_content
        );
        assert_eq!(outcome.content_stream, parsed.content);

        for tag in thinking_tags(session)? {
            assert!(!outcome.reasoning_stream.contains(&tag), "{outcome:?}");
            assert!(!outcome.content_stream.contains(&tag), "{outcome:?}");
        }

        Ok(())
    }
}
