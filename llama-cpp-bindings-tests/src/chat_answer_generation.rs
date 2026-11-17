use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::chat_answer::ChatAnswer;
use crate::classify_sample_loop::ClassifySampleLoop;

fn thinking_tags(session: &ChatSession) -> Result<Vec<String>> {
    let mut tags = session
        .thinking_end_tags()
        .context("the session's thinking end tags must be UTF-8")?;

    tags.push(
        session
            .thinking_start_tag()
            .context("the session's thinking start tag must be UTF-8")?,
    );

    Ok(tags)
}

/// Generates the answer to a chat session from its tokenized prompt, classifying every token.
pub struct ChatAnswerGeneration<'borrow, 'model> {
    pub fixture: &'borrow LlamaFixture<'model>,
    pub session: ChatSession,
    pub sampler: &'borrow mut LlamaSampler,
    pub max_generated_tokens: i32,
}

impl ChatAnswerGeneration<'_, '_> {
    /// # Errors
    ///
    /// Fails when creating the context, tokenizing or decoding the prompt, generating the
    /// answer, or parsing the classifier's final reading fails.
    pub fn run(self) -> Result<ChatAnswer> {
        let Self {
            fixture,
            session,
            sampler,
            max_generated_tokens,
        } = self;
        let mut context = fixture
            .build_context()
            .context("the fixture's context must build")?;
        let prompt_tokens = session
            .tokenize_prompt(fixture.model)
            .context("the session's prompt must tokenize")?;
        let thinking_tags = thinking_tags(&session)?;
        let mut classifier = SampledTokenClassifier::for_chat_session(fixture.model, session)
            .context("the session must read its empty answer")?;
        let mut batch = LlamaBatch::new(ContextLimits::of(&context));

        classifier
            .feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)
            .context("the prompt must fit the batch")?;
        context
            .decode(&batch)
            .context("decoding the prompt must succeed")?;
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

        let parse = classifier
            .latest_reading()
            .context("a generated answer must have been read")?
            .parse()
            .cloned()
            .map_err(Clone::clone)
            .context("the classifier's final reading must parse")?;

        Ok(ChatAnswer {
            outcome,
            usage: classifier.into_usage(),
            thinking_tags,
            parse,
        })
    }
}
