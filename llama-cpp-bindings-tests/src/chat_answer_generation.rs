use anyhow::Result;
use llama_cpp_bindings::BareJsonToolCalls;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_test_harness::LlamaFixture;

use crate::chat_answer::ChatAnswer;
use crate::classify_sample_loop::ClassifySampleLoop;

/// Generates the answer to a chat session from its tokenized prompt, classifying every token.
pub struct ChatAnswerGeneration<'borrow, 'model> {
    pub fixture: &'borrow LlamaFixture<'model>,
    pub session: &'borrow ChatSession<'model>,
    pub sampler: &'borrow mut LlamaSampler,
    pub max_generated_tokens: i32,
}

impl ChatAnswerGeneration<'_, '_> {
    /// # Errors
    ///
    /// Forwards context creation, prompt tokenization, decoding, and sampling errors.
    pub fn run(self) -> Result<ChatAnswer> {
        let Self {
            fixture,
            session,
            sampler,
            max_generated_tokens,
        } = self;
        let mut context = fixture.build_context()?;
        let prompt_tokens = session.tokenize_prompt()?;
        let mut classifier =
            SampledTokenClassifier::for_chat_session(session, BareJsonToolCalls::Detect);
        let mut batch = LlamaBatch::new(ContextLimits::of(&context));

        classifier.feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)?;
        context.decode(&batch)?;
        classifier.commit_prompt_tokens();

        let initial_position = batch.n_tokens();
        let outcome = ClassifySampleLoop {
            classifier: &mut classifier,
            sampler,
            context: &mut context,
            batch: &mut batch,
            initial_position,
            max_generated_tokens,
        }
        .run()?;

        Ok(ChatAnswer {
            outcome,
            usage: classifier.into_usage(),
        })
    }
}
