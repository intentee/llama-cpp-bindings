use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_chat::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::chat_answer::ChatAnswer;
use crate::classify_sample_loop::ClassifySampleLoop;

fn final_parse(classifier: &SampledTokenClassifier<'_>) -> Result<ChatMessageParseOutcome> {
    classifier
        .latest_reading()
        .context("a generated answer must have been read")
        .and_then(|reading| {
            reading
                .parse()
                .cloned()
                .map_err(Clone::clone)
                .context("the classifier's final reading must parse")
        })
}

fn thinking_tags(session: &ChatSession) -> Result<Vec<String>> {
    session
        .thinking_end_tags()
        .context("the session's thinking end tags must be UTF-8")
        .and_then(|mut tags| {
            session
                .thinking_start_tag()
                .context("the session's thinking start tag must be UTF-8")
                .map(|start_tag| {
                    tags.push(start_tag);

                    tags
                })
        })
}

struct PromptedGeneration<'model> {
    context: LlamaContext<'model>,
    batch: LlamaBatch,
    classifier: SampledTokenClassifier<'model>,
}

impl<'model> PromptedGeneration<'model> {
    fn new(
        mut context: LlamaContext<'model>,
        model: &'model LlamaModel,
        session: ChatSession,
    ) -> Result<Self> {
        let mut batch = LlamaBatch::new(context.limits());

        session
            .tokenize_prompt(model)
            .context("the session's prompt must tokenize")
            .and_then(|prompt_tokens| {
                SampledTokenClassifier::for_chat_session(model, session)
                    .context("the session must read its empty answer")
                    .and_then(|mut classifier| {
                        classifier
                            .feed_prompt_sequence_to_batch(&mut batch, &prompt_tokens, 0, false)
                            .context("the prompt must fit the batch")
                            .and_then(|()| {
                                context
                                    .decode(&batch)
                                    .context("decoding the prompt must succeed")
                            })
                            .map(|()| {
                                classifier.commit_prompt_tokens();

                                Self {
                                    context,
                                    batch,
                                    classifier,
                                }
                            })
                    })
            })
    }

    fn answer(
        mut self,
        sampler: &mut LlamaSampler,
        max_generated_tokens: i32,
        thinking_tags: Vec<String>,
    ) -> Result<ChatAnswer> {
        let initial_position = self.batch.n_tokens();

        ClassifySampleLoop {
            classifier: &mut self.classifier,
            sampler,
            context: &mut self.context,
            batch: &mut self.batch,
            initial_position,
            max_generated_tokens,
        }
        .run()
        .and_then(|outcome| {
            final_parse(&self.classifier).map(|parse| ChatAnswer {
                outcome,
                usage: self.classifier.into_usage(),
                thinking_tags,
                parse,
            })
        })
    }
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

        thinking_tags(&session).and_then(|thinking_tags| {
            fixture
                .build_context()
                .context("the fixture's context must build")
                .and_then(|context| PromptedGeneration::new(context, fixture.model, session))
                .and_then(|generation| {
                    generation.answer(sampler, max_generated_tokens, thinking_tags)
                })
        })
    }
}
