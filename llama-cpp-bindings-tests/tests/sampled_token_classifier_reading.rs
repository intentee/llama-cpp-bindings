use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::llama_cpp_error::LlamaCppError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_undeterminable_classifier_has_no_reading_of_the_answer(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut classifier = SampledTokenClassifier::undeterminable(model);

    ingested_generation(&mut classifier, model, "hello")?;

    ensure!(classifier.latest_reading().is_none());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_chat_answer_classifier_parses_the_answer_it_read_last(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;

    ingested_generation(&mut classifier, model, "Hello there.")?;

    ensure!(
        classifier
            .latest_reading()
            .context("the classifier must have read the answer")?
            .parse()
            == Ok(&ChatMessageParseOutcome::Recognized(
                ParsedChatMessage::new("Hello there.".to_owned(), String::new(), Vec::new())
            ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_chat_answer_classifier_reports_a_failed_reading_of_its_empty_answer(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    ensure!(
        SampledTokenClassifier::for_chat_session(model, session).err()
            == Some(LlamaCppError::EmptyAnswerReading(
                ChatError::ParseReported {
                    message: injected_fault_message()?,
                }
            ))
    );

    Ok(())
}
