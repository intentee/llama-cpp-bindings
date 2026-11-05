use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;

use serde::Deserialize;
use serde::Serialize;

use crate::token_usage_error::TokenUsageError;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub input_image_tokens: u64,
    pub input_audio_tokens: u64,
    pub content_tokens: u64,
    pub reasoning_tokens: u64,
    pub tool_call_tokens: u64,
    pub undeterminable_tokens: u64,
}

impl TokenUsage {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            input_image_tokens: 0,
            input_audio_tokens: 0,
            content_tokens: 0,
            reasoning_tokens: 0,
            tool_call_tokens: 0,
            undeterminable_tokens: 0,
        }
    }

    pub const fn record_prompt_tokens(&mut self, count: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(count);
    }

    /// # Errors
    /// Returns [`TokenUsageError::CachedExceedsPrompt`] when the running cached
    /// total would exceed [`Self::prompt_tokens`].
    pub const fn record_cached_prompt_tokens(&mut self, count: u64) -> Result<(), TokenUsageError> {
        let next = self.cached_prompt_tokens.saturating_add(count);

        if next > self.prompt_tokens {
            return Err(TokenUsageError::CachedExceedsPrompt {
                cached_after: next,
                prompt: self.prompt_tokens,
            });
        }

        self.cached_prompt_tokens = next;

        Ok(())
    }

    pub const fn record_input_image_tokens(&mut self, count: u64) {
        self.input_image_tokens = self.input_image_tokens.saturating_add(count);
    }

    pub const fn record_input_audio_tokens(&mut self, count: u64) {
        self.input_audio_tokens = self.input_audio_tokens.saturating_add(count);
    }

    pub const fn record_content_token(&mut self) {
        self.content_tokens = self.content_tokens.saturating_add(1);
    }

    pub const fn record_reasoning_token(&mut self) {
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
    }

    pub const fn record_tool_call_token(&mut self) {
        self.tool_call_tokens = self.tool_call_tokens.saturating_add(1);
    }

    pub const fn record_undeterminable_token(&mut self) {
        self.undeterminable_tokens = self.undeterminable_tokens.saturating_add(1);
    }

    #[must_use]
    pub const fn completion_tokens(&self) -> u64 {
        self.content_tokens
            .saturating_add(self.reasoning_tokens)
            .saturating_add(self.tool_call_tokens)
            .saturating_add(self.undeterminable_tokens)
    }

    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens())
    }
}

impl Add for TokenUsage {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl Add<&Self> for TokenUsage {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self += other;
        self
    }
}

impl AddAssign for TokenUsage {
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl AddAssign<&Self> for TokenUsage {
    fn add_assign(&mut self, other: &Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.cached_prompt_tokens = self
            .cached_prompt_tokens
            .saturating_add(other.cached_prompt_tokens);
        self.input_image_tokens = self
            .input_image_tokens
            .saturating_add(other.input_image_tokens);
        self.input_audio_tokens = self
            .input_audio_tokens
            .saturating_add(other.input_audio_tokens);
        self.content_tokens = self.content_tokens.saturating_add(other.content_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        self.tool_call_tokens = self.tool_call_tokens.saturating_add(other.tool_call_tokens);
        self.undeterminable_tokens = self
            .undeterminable_tokens
            .saturating_add(other.undeterminable_tokens);
    }
}

impl Sum for TokenUsage {
    fn sum<TIter: Iterator<Item = Self>>(iter: TIter) -> Self {
        iter.fold(Self::new(), |acc, item| acc + item)
    }
}

impl<'usage> Sum<&'usage Self> for TokenUsage {
    fn sum<TIter: Iterator<Item = &'usage Self>>(iter: TIter) -> Self {
        iter.fold(Self::new(), |acc, item| acc + item)
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::TokenUsage;
    use super::TokenUsageError;

    const LEFT_USAGE: TokenUsage = TokenUsage {
        prompt_tokens: 20,
        cached_prompt_tokens: 2,
        input_image_tokens: 3,
        input_audio_tokens: 4,
        content_tokens: 5,
        reasoning_tokens: 6,
        tool_call_tokens: 7,
        undeterminable_tokens: 8,
    };

    const RIGHT_USAGE: TokenUsage = TokenUsage {
        prompt_tokens: 200,
        cached_prompt_tokens: 20,
        input_image_tokens: 30,
        input_audio_tokens: 40,
        content_tokens: 50,
        reasoning_tokens: 60,
        tool_call_tokens: 70,
        undeterminable_tokens: 80,
    };

    const COMBINED_USAGE: TokenUsage = TokenUsage {
        prompt_tokens: 220,
        cached_prompt_tokens: 22,
        input_image_tokens: 33,
        input_audio_tokens: 44,
        content_tokens: 55,
        reasoning_tokens: 66,
        tool_call_tokens: 77,
        undeterminable_tokens: 88,
    };

    #[test]
    fn new_starts_with_all_counters_at_zero() {
        let usage = TokenUsage::new();

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.cached_prompt_tokens, 0);
        assert_eq!(usage.input_image_tokens, 0);
        assert_eq!(usage.input_audio_tokens, 0);
        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 0);
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

        assert_eq!(usage.prompt_tokens, 7);
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

        assert_eq!(usage.cached_prompt_tokens, 7);
    }

    #[test]
    fn caching_every_prompt_token_is_accepted() {
        let mut usage = TokenUsage::new();
        usage.record_prompt_tokens(5);
        usage
            .record_cached_prompt_tokens(5)
            .expect("5 cached == 5 prompt is valid (boundary)");

        assert_eq!(usage.cached_prompt_tokens, 5);
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
        assert_eq!(usage.cached_prompt_tokens, 0);
    }

    #[test]
    fn record_input_image_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(5);
        usage.record_input_image_tokens(3);

        assert_eq!(usage.input_image_tokens, 8);
    }

    #[test]
    fn record_input_audio_tokens_accumulates() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(2);
        usage.record_input_audio_tokens(9);

        assert_eq!(usage.input_audio_tokens, 11);
    }

    #[test]
    fn input_image_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_image_tokens(40);

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn input_audio_tokens_do_not_contribute_to_prompt_or_completion() {
        let mut usage = TokenUsage::new();
        usage.record_input_audio_tokens(40);

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens(), 0);
    }

    #[test]
    fn record_content_token_increments_only_content() {
        let mut usage = TokenUsage::new();
        usage.record_content_token();

        assert_eq!(usage.content_tokens, 1);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn record_reasoning_token_increments_only_reasoning() {
        let mut usage = TokenUsage::new();
        usage.record_reasoning_token();

        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 1);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn record_tool_call_token_increments_only_tool_call() {
        let mut usage = TokenUsage::new();
        usage.record_tool_call_token();

        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 1);
        assert_eq!(usage.undeterminable_tokens, 0);
    }

    #[test]
    fn record_undeterminable_token_increments_only_undeterminable() {
        let mut usage = TokenUsage::new();
        usage.record_undeterminable_token();

        assert_eq!(usage.content_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
        assert_eq!(usage.tool_call_tokens, 0);
        assert_eq!(usage.undeterminable_tokens, 1);
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
