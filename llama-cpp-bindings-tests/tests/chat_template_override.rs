use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use std::fs;
use std::path::PathBuf;

use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::get_weather_tools::get_weather_tools;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::paris_weather_tool_call::paris_weather_tool_call;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_message::user_message;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_chat::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_chat::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_chat::chat_session_options::ChatSessionOptions;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::chat_templates::ChatTemplates;
use llama_cpp_model::llama_chat_template::LlamaChatTemplate;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
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

fn weather_session(model: &LlamaModel, templates: &ChatTemplates) -> Result<ChatSession> {
    Ok(ChatSession::new(
        model,
        templates,
        &[user_message("What is the weather in Paris?")],
        &get_weather_tools(),
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
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_served_with_the_llama_cpp_template_parses_its_tool_call(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let templates = model.chat_templates_with_override(&llama_cpp_deepseek_r1_template()?)?;
    let session = weather_session(model, &templates)?;

    ensure!(
        recognized_answer(
            &session,
            &format!("{REASONING}{R1_WEATHER_TOOL_CALL}"),
            ChatOutputCompleteness::Finished
        )? == ParsedChatMessage::new(
            String::new(),
            "\nThe user wants the weather in Paris.".to_owned(),
            vec![paris_weather_tool_call()]
        )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_served_with_the_llama_cpp_template_streams_its_tool_call_tokens_as_a_tool_call(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let templates = model.chat_templates_with_override(&llama_cpp_deepseek_r1_template()?)?;
    let session = weather_session(model, &templates)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;

    ingested_generation(
        &mut classifier,
        model,
        &format!("{REASONING}{R1_WEATHER_TOOL_CALL}"),
    )?;

    ensure!(
        [
            classifier.usage().reasoning,
            classifier.usage().content,
            classifier.usage().tool_call,
        ] == [
            token_count(model, REASONING)?,
            0,
            token_count(model, R1_WEATHER_TOOL_CALL)?.checked_add(END_OF_GENERATION_TOKENS).context("`token_count(fixture.model, R1_WEATHER_TOOL_CALL)? + END_OF_GENERATION_TOKENS` must not overflow")?,
        ]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_empty_template_override_is_refused(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        model
            .chat_templates_with_override(&LlamaChatTemplate::new("")?)
            .err()
            == Some(ModelError::EmptyTemplateOverride)
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 1024,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_template_override_llama_cpp_cannot_parse_reports_its_parse_error(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(model.chat_templates_with_override(
            &LlamaChatTemplate::new("{{invalid jinja template")?
        )
        .err() == Some(ModelError::ChatTemplatesReported {
            message: "parser: Parser Error: Expected }} (Got jinja)\n...{{invalid jinja template...\n             ^".to_owned(),
        }));

    Ok(())
}
