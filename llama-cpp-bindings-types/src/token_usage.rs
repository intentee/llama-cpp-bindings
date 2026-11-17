use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;

use serde::Deserialize;
use serde::Serialize;

use crate::token_usage_error::TokenUsageError;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenUsage {
    #[serde(rename = "prompt_tokens")]
    pub prompt: u64,
    #[serde(rename = "cached_prompt_tokens")]
    pub cached_prompt: u64,
    #[serde(rename = "input_image_tokens")]
    pub input_image: u64,
    #[serde(rename = "input_audio_tokens")]
    pub input_audio: u64,
    #[serde(rename = "content_tokens")]
    pub content: u64,
    #[serde(rename = "reasoning_tokens")]
    pub reasoning: u64,
    #[serde(rename = "tool_call_tokens")]
    pub tool_call: u64,
    #[serde(rename = "undeterminable_tokens")]
    pub undeterminable: u64,
}

impl TokenUsage {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prompt: 0,
            cached_prompt: 0,
            input_image: 0,
            input_audio: 0,
            content: 0,
            reasoning: 0,
            tool_call: 0,
            undeterminable: 0,
        }
    }

    #[must_use]
    pub const fn completion_tokens(&self) -> u64 {
        self.content
            .saturating_add(self.reasoning)
            .saturating_add(self.tool_call)
            .saturating_add(self.undeterminable)
    }

    /// # Errors
    /// Returns [`TokenUsageError::CachedExceedsPrompt`] when the running cached
    /// total would exceed [`Self::prompt`].
    pub const fn record_cached_prompt_tokens(&mut self, count: u64) -> Result<(), TokenUsageError> {
        let next = self.cached_prompt.saturating_add(count);

        if next > self.prompt {
            return Err(TokenUsageError::CachedExceedsPrompt {
                cached_after: next,
                prompt: self.prompt,
            });
        }

        self.cached_prompt = next;

        Ok(())
    }

    pub const fn record_content_token(&mut self) {
        self.content = self.content.saturating_add(1);
    }

    pub const fn record_input_audio_tokens(&mut self, count: u64) {
        self.input_audio = self.input_audio.saturating_add(count);
    }

    pub const fn record_input_image_tokens(&mut self, count: u64) {
        self.input_image = self.input_image.saturating_add(count);
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.prompt = self.prompt.saturating_add(count);
    }

    pub const fn record_reasoning_token(&mut self) {
        self.reasoning = self.reasoning.saturating_add(1);
    }

    pub const fn record_tool_call_token(&mut self) {
        self.tool_call = self.tool_call.saturating_add(1);
    }

    pub const fn record_undeterminable_token(&mut self) {
        self.undeterminable = self.undeterminable.saturating_add(1);
    }

    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.prompt.saturating_add(self.completion_tokens())
    }

    const fn accumulate(
        &mut self,
        &Self {
            prompt,
            cached_prompt,
            input_image,
            input_audio,
            content,
            reasoning,
            tool_call,
            undeterminable,
        }: &Self,
    ) {
        self.prompt = self.prompt.saturating_add(prompt);
        self.cached_prompt = self.cached_prompt.saturating_add(cached_prompt);
        self.input_image = self.input_image.saturating_add(input_image);
        self.input_audio = self.input_audio.saturating_add(input_audio);
        self.content = self.content.saturating_add(content);
        self.reasoning = self.reasoning.saturating_add(reasoning);
        self.tool_call = self.tool_call.saturating_add(tool_call);
        self.undeterminable = self.undeterminable.saturating_add(undeterminable);
    }
}

impl Add for TokenUsage {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self.accumulate(&other);
        self
    }
}

impl Add<&Self> for TokenUsage {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.accumulate(other);
        self
    }
}

impl AddAssign for TokenUsage {
    fn add_assign(&mut self, other: Self) {
        self.accumulate(&other);
    }
}

impl AddAssign<&Self> for TokenUsage {
    fn add_assign(&mut self, other: &Self) {
        self.accumulate(other);
    }
}

impl Sum for TokenUsage {
    fn sum<TIter: Iterator<Item = Self>>(iter: TIter) -> Self {
        iter.fold(Self::new(), Add::add)
    }
}

impl<'usage> Sum<&'usage Self> for TokenUsage {
    fn sum<TIter: Iterator<Item = &'usage Self>>(iter: TIter) -> Self {
        iter.fold(Self::new(), Add::add)
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::TokenUsage;
    use super::TokenUsageError;

    const LEFT_USAGE: TokenUsage = TokenUsage {
        prompt: 20,
        cached_prompt: 2,
        input_image: 3,
        input_audio: 4,
        content: 5,
        reasoning: 6,
        tool_call: 7,
        undeterminable: 8,
    };

    const RIGHT_USAGE: TokenUsage = TokenUsage {
        prompt: 200,
        cached_prompt: 20,
        input_image: 30,
        input_audio: 40,
        content: 50,
        reasoning: 60,
        tool_call: 70,
        undeterminable: 80,
    };

    const COMBINED_USAGE: TokenUsage = TokenUsage {
        prompt: 220,
        cached_prompt: 22,
        input_image: 33,
        input_audio: 44,
        content: 55,
        reasoning: 66,
        tool_call: 77,
        undeterminable: 88,
    };

    #[test]
    fn new_starts_with_all_counters_at_zero() {
        let usage = TokenUsage::new();

        assert_eq!(usage.prompt, 0);
        assert_eq!(usage.cached_prompt, 0);
        assert_eq!(usage.input_image, 0);
        assert_eq!(usage.input_audio, 0);
        assert_eq!(usage.content, 0);
        assert_eq!(usage.reasoning, 0);
        assert_eq!(usage.tool_call, 0);
        assert_eq!(usage.undeterminable, 0);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(TokenUsage::default(), TokenUsage::new());
    }

    #[test]
    fn completion_is_zero_when_no_events_recorded() {
        assert_eq!(TokenUsage::new().completion_tokens(), 0);
    }

    #[test]
    fn total_equals_prompt_plus_completion() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(3);
        usage.record_content_token();
        usage.record_reasoning_token();

        assert_eq!(usage.total_tokens(), 5);
    }

    #[test]
    fn record_prompt_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(3);
        usage.record_prompt_tokens(4);

        assert_eq!(usage.prompt, 7);
    }

    #[test]
    fn record_cached_below_prompt_succeeds_and_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(10);
        usage
            .record_cached_prompt_tokens(3)
            .expect("3 cached <= 10 prompt is valid");
        usage
            .record_cached_prompt_tokens(4)
            .expect("3+4 cached <= 10 prompt is valid");

        assert_eq!(usage.cached_prompt, 7);
    }

    #[test]
    fn caching_every_prompt_token_is_accepted() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(5);
        usage
            .record_cached_prompt_tokens(5)
            .expect("5 cached == 5 prompt is valid (boundary)");

        assert_eq!(usage.cached_prompt, 5);
    }

    #[test]
    fn record_cached_above_prompt_returns_error_and_does_not_mutate() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(2);

        let result = usage.record_cached_prompt_tokens(3);

        assert_eq!(
            result,
            Err(TokenUsageError::CachedExceedsPrompt {
                cached_after: 3,
                prompt: 2,
            })
        );
        assert_eq!(usage.cached_prompt, 0);
    }

    #[test]
    fn record_input_image_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(5);
        usage.record_input_image_tokens(3);

        assert_eq!(usage.input_image, 8);
    }

    #[test]
    fn record_input_audio_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(2);
        usage.record_input_audio_tokens(9);

        assert_eq!(usage.input_audio, 11);
    }

    #[test]
    fn input_image_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(40);

        assert_eq!(usage.prompt, 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn input_audio_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(40);

        assert_eq!(usage.prompt, 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn record_content_token_increments_only_content() {
        let mut usage = TokenUsage::new();
        usage.record_content_token();

        assert_eq!(usage.content, 1);
        assert_eq!(usage.reasoning, 0);
        assert_eq!(usage.tool_call, 0);
        assert_eq!(usage.undeterminable, 0);
    }

    #[test]
    fn record_reasoning_token_increments_only_reasoning() {
        let mut usage = TokenUsage::new();
        usage.record_reasoning_token();

        assert_eq!(usage.content, 0);
        assert_eq!(usage.reasoning, 1);
        assert_eq!(usage.tool_call, 0);
        assert_eq!(usage.undeterminable, 0);
    }

    #[test]
    fn record_tool_call_token_increments_only_tool_call() {
        let mut usage = TokenUsage::new();
        usage.record_tool_call_token();

        assert_eq!(usage.content, 0);
        assert_eq!(usage.reasoning, 0);
        assert_eq!(usage.tool_call, 1);
        assert_eq!(usage.undeterminable, 0);
    }

    #[test]
    fn record_undeterminable_token_increments_only_undeterminable() {
        let mut usage = TokenUsage::new();
        usage.record_undeterminable_token();

        assert_eq!(usage.content, 0);
        assert_eq!(usage.reasoning, 0);
        assert_eq!(usage.tool_call, 0);
        assert_eq!(usage.undeterminable, 1);
    }

    #[test]
    fn completion_tokens_sums_every_output_kind() {
        let mut usage = TokenUsage::new();
        usage.record_content_token();
        usage.record_content_token();
        usage.record_reasoning_token();
        usage.record_tool_call_token();
        usage.record_undeterminable_token();

        assert_eq!(usage.completion_tokens(), 5);
    }

    #[test]
    fn add_combines_every_field() {
        assert_eq!(LEFT_USAGE + RIGHT_USAGE, COMBINED_USAGE);
    }

    #[test]
    fn add_assign_matches_add() {
        let mut left = TokenUsage::new();
        left.record_prompt_tokens(2);
        left.record_content_token();

        let mut right = TokenUsage::new();
        right.record_prompt_tokens(3);
        right.record_reasoning_token();

        let added = left + right;

        let mut accumulating = left;
        accumulating += right;

        assert_eq!(accumulating, added);
    }

    #[test]
    fn add_ref_combines_every_field() {
        let right_usage = &RIGHT_USAGE;

        assert_eq!(LEFT_USAGE + right_usage, COMBINED_USAGE);
    }

    #[test]
    fn sum_over_iter_matches_repeated_add_assign() {
        let mut prompt_only = TokenUsage::new();
        prompt_only.record_prompt_tokens(1);

        let mut prompt_with_content = TokenUsage::new();
        prompt_with_content.record_prompt_tokens(2);
        prompt_with_content.record_content_token();

        let mut prompt_with_reasoning = TokenUsage::new();
        prompt_with_reasoning.record_prompt_tokens(4);
        prompt_with_reasoning.record_reasoning_token();

        let summed: TokenUsage = [prompt_only, prompt_with_content, prompt_with_reasoning]
            .into_iter()
            .sum();
        let summed_ref: TokenUsage = [&prompt_only, &prompt_with_content, &prompt_with_reasoning]
            .into_iter()
            .sum();

        let mut acc = TokenUsage::new();
        acc += prompt_only;
        acc += prompt_with_content;
        acc += prompt_with_reasoning;

        assert_eq!(summed, acc);
        assert_eq!(summed_ref, acc);
    }

    #[test]
    fn sum_over_empty_iter_returns_default() {
        let summed: TokenUsage = iter::empty::<TokenUsage>().sum();

        assert_eq!(summed, TokenUsage::default());
    }
}
