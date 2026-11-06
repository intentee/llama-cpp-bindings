use anyhow::Result;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings_tests::user_message::user_message;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_types::chat_message::ChatMessage;
use llama_cpp_bindings_types::chat_message_content::ChatMessageContent;
use llama_cpp_bindings_types::chat_message_tool_call::ChatMessageToolCall;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace(
        "unsloth/Ministral-3-14B-Reasoning-2512-GGUF",
        "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn chat_session_renders_the_user_message_into_the_prompt(fixture: &LlamaFixture<'_>) -> Result<()> {
    let prompt = user_turn_chat_session(fixture.model, "hello", &[], true)?.prompt()?;

    assert!(
        prompt.contains("hello"),
        "the model's built-in chat template must render the user message content through the \
         engine; got: {prompt:?}"
    );
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn chat_session_renders_long_messages_in_full(fixture: &LlamaFixture<'_>) -> Result<()> {
    let long_content = "a".repeat(2000);
    let prompt = user_turn_chat_session(fixture.model, &long_content, &[], true)?.prompt()?;

    assert!(
        prompt.contains(&long_content),
        "a long user message must be rendered in full by the engine without truncation"
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
fn qwen35_generation_prompt_opens_the_assistant_turn_with_thinking(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], true)?;

    assert_eq!(
        session.generation_prompt()?,
        "<|im_start|>assistant\n<think>\n"
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
fn qwen35_omitted_assistant_turn_renders_only_the_messages(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        fixture.model.chat_templates()?,
        &[user_message("hello")],
        &[],
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::Omitted,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;

    assert_eq!(session.prompt()?, "<|im_start|>user\nhello<|im_end|>\n");

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
fn qwen35_renders_a_tool_call_history_with_its_reasoning_and_tool_response(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        fixture.model.chat_templates()?,
        &[
            user_message("What is the weather in Paris?"),
            ChatMessage::AssistantToolCalls {
                content: ChatMessageContent::Text(String::new()),
                reasoning_content: Some("I need the weather tool.".to_owned()),
                tool_calls: vec![ChatMessageToolCall {
                    id: "call_weather".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: r#"{"location":"Paris"}"#.to_owned(),
                }],
            },
            ChatMessage::ToolResult {
                content: ChatMessageContent::Text("sunny".to_owned()),
                tool_call_id: "call_weather".to_owned(),
            },
        ],
        &[],
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::Omitted,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;

    assert_eq!(
        session.prompt()?,
        concat!(
            "<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n",
            "<|im_start|>assistant\n<think>\nI need the weather tool.\n</think>\n\n",
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n",
            "</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n",
        )
    );

    Ok(())
}
