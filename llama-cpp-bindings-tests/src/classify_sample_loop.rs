use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::classified_sample::ClassifiedSample;
use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_sampler::llama_sampler::LlamaSampler;

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
    pub fn run(mut self) -> Result<ClassifySampleLoopOutcome> {
        let mut ingest_outcomes = Vec::new();
        let positions = self.initial_position
            ..self
                .initial_position
                .saturating_add(self.max_generated_tokens);
        let first_stop = positions
            .map(|position| self.generate_at(position, &mut ingest_outcomes))
            .find(|generated| generated.as_ref().ok() != Some(&GenerationProgress::Continues));

        first_stop
            .unwrap_or(Ok(GenerationProgress::Continues))
            .and_then(|progress| self.settle(progress, &mut ingest_outcomes))
            .map(|mut outcome| {
                for ingest_outcome in &ingest_outcomes {
                    outcome.record_outcome(ingest_outcome);
                }

                outcome
            })
    }

    fn decode_at(&mut self, position: i32, token: LlamaToken) -> Result<()> {
        self.batch.clear();
        self.batch
            .add(SampledToken::Content(token), position, &[0], true)
            .context("the sampled token must fit the batch")
            .and_then(|()| {
                self.context
                    .decode(self.batch)
                    .context("decoding the sampled token must succeed")
            })
    }

    fn generate_at(
        &mut self,
        position: i32,
        ingest_outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<GenerationProgress> {
        self.classifier
            .sample(
                self.sampler,
                self.context,
                self.batch.n_tokens().saturating_sub(1),
                ingest_outcomes,
            )
            .context("sampling and classifying the next token must succeed")
            .and_then(|ClassifiedSample { token, progress }| match progress {
                GenerationProgress::Ended => Ok(progress),
                GenerationProgress::Continues => self.decode_at(position, token).map(|()| progress),
            })
    }

    fn settle(
        &mut self,
        progress: GenerationProgress,
        ingest_outcomes: &mut Vec<IngestOutcome>,
    ) -> Result<ClassifySampleLoopOutcome> {
        let mut outcome = ClassifySampleLoopOutcome::default();

        match progress {
            GenerationProgress::Ended => {
                outcome.record_end_of_generation(self.classifier.current_section());

                Ok(outcome)
            }
            GenerationProgress::Continues => self
                .classifier
                .finish(ingest_outcomes)
                .context("finishing the classification must succeed")
                .map(|()| outcome),
        }
    }
}
