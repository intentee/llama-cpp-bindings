use anyhow::Result;
use llama_cpp_bindings::GenerationProgress;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;

use crate::classify_sample_loop_outcome::ClassifySampleLoopOutcome;

pub struct ClassifySampleLoop<'borrow, 'session, 'model> {
    pub classifier: &'borrow mut SampledTokenClassifier<'session, 'model>,
    pub sampler: &'borrow mut LlamaSampler,
    pub context: &'borrow mut LlamaContext<'model>,
    pub batch: &'borrow mut LlamaBatch,
    pub initial_position: i32,
    pub max_generated_tokens: i32,
}

impl ClassifySampleLoop<'_, '_, '_> {
    /// # Errors
    /// Forwards [`SampledTokenClassifier::sample`] / [`SampledTokenClassifier::finish`] /
    /// [`LlamaContext::decode`] / [`LlamaBatch::add`] errors verbatim. Stops on the end of generation, on
    /// `max_generated_tokens` exhaustion, or on the first error.
    pub fn run(self) -> Result<ClassifySampleLoopOutcome> {
        let mut outcome = ClassifySampleLoopOutcome::default();
        let mut ingest_outcomes = Vec::new();
        let mut position = self.initial_position;
        let max_position = position + self.max_generated_tokens;

        while position < max_position {
            let sampled = self.classifier.sample(
                self.sampler,
                self.context,
                self.batch.n_tokens() - 1,
                &mut ingest_outcomes,
            )?;

            if sampled.progress == GenerationProgress::Ended {
                outcome.record_end_of_generation(self.classifier.current_section());

                break;
            }

            self.batch.clear();
            self.batch
                .add(&SampledToken::Content(sampled.token), position, &[0], true)?;
            position += 1;

            self.context.decode(self.batch)?;
        }

        self.classifier.finish(&mut ingest_outcomes)?;

        for ingest_outcome in &ingest_outcomes {
            outcome.record_outcome(ingest_outcome);
        }

        Ok(outcome)
    }
}
