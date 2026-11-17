use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;

use crate::classify_sample_loop_outcome::ClassifySampleLoopOutcome;

pub struct ClassifySampleLoop<'borrow, 'model> {
    pub classifier: &'borrow mut SampledTokenClassifier<'model>,
    pub sampler: &'borrow mut LlamaSampler,
    pub context: &'borrow mut LlamaContext<'model>,
    pub batch: &'borrow mut LlamaBatch,
    pub initial_position: i32,
    pub max_generated_tokens: i32,
}

impl ClassifySampleLoop<'_, '_> {
    /// Samples and classifies tokens until the end of generation or until
    /// `max_generated_tokens` are generated, settling the pending tokens of a generation the
    /// limit stops; an end-of-generation token settles them itself.
    ///
    /// # Errors
    ///
    /// Fails when sampling, classifying, batching, or decoding a token fails.
    pub fn run(self) -> Result<ClassifySampleLoopOutcome> {
        let mut outcome = ClassifySampleLoopOutcome::default();
        let mut ingest_outcomes = Vec::new();
        let mut position = self.initial_position;
        let max_position = position
            .checked_add(self.max_generated_tokens)
            .context("the last generated position must fit a llama.cpp position")?;

        while position < max_position {
            let last_batch_index = self
                .batch
                .n_tokens()
                .checked_sub(1)
                .context("the batch must hold the token to sample after")?;
            let sampled = self
                .classifier
                .sample(
                    self.sampler,
                    self.context,
                    last_batch_index,
                    &mut ingest_outcomes,
                )
                .context("sampling and classifying the next token must succeed")?;

            if sampled.progress == GenerationProgress::Ended {
                outcome.record_end_of_generation(self.classifier.current_section())?;

                break;
            }

            self.batch.clear();
            self.batch
                .add(SampledToken::Content(sampled.token), position, &[0], true)
                .context("the sampled token must fit the batch")?;
            position = position
                .checked_add(1)
                .context("the next position must fit a llama.cpp position")?;

            self.context
                .decode(self.batch)
                .context("decoding the sampled token must succeed")?;
        }

        if !outcome.eog_seen {
            self.classifier
                .finish(&mut ingest_outcomes)
                .context("finishing the classification must succeed")?;
        }

        for ingest_outcome in &ingest_outcomes {
            outcome.record_outcome(ingest_outcome)?;
        }

        Ok(outcome)
    }
}
