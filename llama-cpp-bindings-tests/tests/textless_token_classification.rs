use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
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
fn a_token_that_completes_no_character_is_classified_without_reading_the_answer_again(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(model, "hello", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let hieroglyph_tokens = model.str_to_token("𓀀", AddBos::Never, ParseSpecialTokens::Never)?;
    let leading_byte_token = *hieroglyph_tokens
        .first()
        .context("the hieroglyph must tokenize")?;

    ensure!(hieroglyph_tokens.len() > 1);

    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS,
    );

    ensure!(
        classifier.ingest(leading_byte_token, &mut Vec::new()) == Ok(GenerationProgress::Continues)
    );

    Ok(())
}
