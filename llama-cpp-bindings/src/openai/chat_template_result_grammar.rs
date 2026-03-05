use std::collections::HashSet;

use crate::model::{AddBos, ChatTemplateResult, GrammarTriggerType, LlamaModel};
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

use super::grammar_sampler_error::GrammarSamplerError;

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());

    for character in value.chars() {
        match character {
            '.' | '^' | '$' | '|' | '(' | ')' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '\\' => {
                escaped.push('\\');
                escaped.push(character);
            }
            _ => escaped.push(character),
        }
    }

    escaped
}

fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }

    let mut anchored = String::new();

    if !pattern.starts_with('^') {
        anchored.push('^');
    }

    anchored.push_str(pattern);

    if !pattern.ends_with('$') {
        anchored.push('$');
    }

    anchored
}

impl ChatTemplateResult {
    /// Builds a grammar sampler from this template result's grammar and trigger configuration.
    ///
    /// Returns `None` if no grammar is present. The returned `HashSet` contains preserved
    /// token IDs that should be decoded with special token handling.
    ///
    /// # Errors
    /// Returns an error if trigger processing or grammar sampler initialization fails.
    pub fn build_grammar_sampler(
        &self,
        model: &LlamaModel,
    ) -> Result<(Option<LlamaSampler>, HashSet<LlamaToken>), GrammarSamplerError> {
        let mut preserved = HashSet::new();

        for token_str in &self.preserved_tokens {
            let tokens = model
                .str_to_token(token_str, AddBos::Never)
                .map_err(|error| GrammarSamplerError::TokenizationFailed(error.to_string()))?;

            if tokens.len() == 1 {
                preserved.insert(tokens[0]);
            }
        }

        let Some(grammar) = self.grammar.as_deref() else {
            return Ok((None, preserved));
        };

        let grammar_sampler = if self.grammar_lazy {
            if self.grammar_triggers.is_empty() {
                return Err(GrammarSamplerError::MissingTriggers);
            }

            let mut trigger_patterns = Vec::new();
            let mut trigger_tokens = Vec::new();

            for trigger in &self.grammar_triggers {
                match trigger.trigger_type {
                    GrammarTriggerType::Token => {
                        if let Some(token) = trigger.token {
                            trigger_tokens.push(token);
                        }
                    }
                    GrammarTriggerType::Word => {
                        let tokens =
                            model
                                .str_to_token(&trigger.value, AddBos::Never)
                                .map_err(|error| {
                                    GrammarSamplerError::TokenizationFailed(error.to_string())
                                })?;

                        if tokens.len() == 1 {
                            if !preserved.contains(&tokens[0]) {
                                return Err(GrammarSamplerError::TriggerWordNotPreserved(
                                    trigger.value.clone(),
                                ));
                            }
                            trigger_tokens.push(tokens[0]);
                        } else {
                            trigger_patterns.push(regex_escape(&trigger.value));
                        }
                    }
                    GrammarTriggerType::Pattern => {
                        trigger_patterns.push(trigger.value.clone());
                    }
                    GrammarTriggerType::PatternFull => {
                        trigger_patterns.push(anchor_pattern(&trigger.value));
                    }
                }
            }

            LlamaSampler::grammar_lazy_patterns(
                model,
                grammar,
                "root",
                &trigger_patterns,
                &trigger_tokens,
            )?
        } else {
            LlamaSampler::grammar(model, grammar, "root")?
        };

        Ok((Some(grammar_sampler), preserved))
    }
}
