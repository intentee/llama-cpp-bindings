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
    /// # Panics
    ///
    /// Panics when creating the context, tokenizing or decoding the prompt, or generating the
    /// answer fails.
    #[must_use]
    pub fn run(self) -> ChatAnswer {
        let Self {
            fixture,
            session,
            sampler,
            max_generated_tokens,
        } = self;
        let mut context = fixture
            .build_context()
            .expect("the fixture's context must build");
        let prompt_tokens = session
            .tokenize_prompt()
            .expect("the session's prompt must tokenize");
        let mut classifier =
            SampledTokenClassifier::for_chat_session(session, BareJsonToolCalls::Detect);
        let mut batch = LlamaBatch::new(ContextLimits::of(&context));

        classifier
            .feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)
            .expect("the prompt must fit the batch");
        context
            .decode(&batch)
            .expect("decoding the prompt must succeed");
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
        .run();

        ChatAnswer {
            outcome,
            usage: classifier.into_usage(),
        }
    }
}
