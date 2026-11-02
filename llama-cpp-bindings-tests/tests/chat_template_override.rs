use std::fs;
use std::path::PathBuf;

use anyhow::Result;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_templates::ChatTemplates;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::error::chat_templates_error::ChatTemplatesError;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::model::llama_chat_template::LlamaChatTemplate;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::get_weather_tools_json::GET_WEATHER_TOOLS_JSON;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::paris_weather_tool_call::paris_weather_tool_call;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const REASONING: &str = "The user wants the weather in Paris.</think>";
const R1_WEATHER_TOOL_CALL: &str = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"location\": \"Paris\"}```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
const END_OF_GENERATION_TOKENS: u64 = 1;

fn llama_cpp_deepseek_r1_template() -> Result<LlamaChatTemplate> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../llama-cpp-bindings-sys/llama.cpp/models/templates/llama-cpp-deepseek-r1.jinja");

    Ok(LlamaChatTemplate::new(&fs::read_to_string(path)?)?)
}

fn weather_session<'model>(
    model: &'model LlamaModel,
    templates: &ChatTemplates,
) -> Result<ChatSession<'model>> {
    Ok(ChatSession::new(
        model,
        templates,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            "What is the weather in Paris?".to_owned(),
        )?],
        GET_WEATHER_TOOLS_JSON,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?)
}

fn token_count(model: &LlamaModel, text: &str) -> Result<u64> {
    Ok(u64::try_from(
        model
            .str_to_token(text, AddBos::Never, ParseSpecialTokens::Always)?
            .len(),
    )?)
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_served_with_the_llama_cpp_template_parses_its_tool_call(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let templates =
        ChatTemplates::with_override(fixture.model, &llama_cpp_deepseek_r1_template()?)?;
    let session = weather_session(fixture.model, &templates)?;

    assert_eq!(
        recognized_answer(
            &session,
            &format!("{REASONING}{R1_WEATHER_TOOL_CALL}"),
            ChatOutputCompleteness::Finished
        ),
        ParsedChatMessage::new(
            String::new(),
            "\nThe user wants the weather in Paris.".to_owned(),
            vec![paris_weather_tool_call()]
        )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_served_with_the_llama_cpp_template_streams_its_tool_call_tokens_as_a_tool_call(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let templates =
        ChatTemplates::with_override(fixture.model, &llama_cpp_deepseek_r1_template()?)?;
    let session = weather_session(fixture.model, &templates)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(&session);

    ingested_generation(
        &mut classifier,
        fixture.model,
        &format!("{REASONING}{R1_WEATHER_TOOL_CALL}"),
    );

    assert_eq!(
        [
            classifier.usage().reasoning_tokens,
            classifier.usage().content_tokens,
            classifier.usage().tool_call_tokens,
        ],
        [
            token_count(fixture.model, REASONING)?,
            0,
            token_count(fixture.model, R1_WEATHER_TOOL_CALL)? + END_OF_GENERATION_TOKENS,
        ]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_empty_template_override_is_refused(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(
        ChatTemplates::with_override(fixture.model, &LlamaChatTemplate::new("")?).err(),
        Some(ChatTemplatesError::EmptyTemplateOverride)
    );

    Ok(())
}
