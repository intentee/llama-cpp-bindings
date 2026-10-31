use anyhow::Result;
use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::chat_answer_generation::ChatAnswerGeneration;
use llama_cpp_bindings_tests::closed_reasoning_chat_session::closed_reasoning_chat_session;
use llama_cpp_bindings_tests::penalized_nucleus_sampler::penalized_nucleus_sampler;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const ARITHMETIC_QUESTION: &str = "What is 2 + 2?";
const SINGLE_WORD_INSTRUCTION: &str = "Reply with the single word: four. Do not explain.";
const MAX_ANSWER_TOKENS: i32 = 200;
const MAX_REASONING_ANSWER_TOKENS: i32 = 512;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn deepseek_r1_8b_answers_without_reasoning_after_its_reasoning_is_closed(
    fixture: &LlamaFixture<'_>,
) {
    let session = closed_reasoning_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]");
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    answer.assert_content_without_reasoning(&session);
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn deepseek_r1_8b_reasoning_streams_match_the_parsed_answer_when_thinking_is_enabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", true)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_REASONING_ANSWER_TOKENS,
    }
    .run();

    answer.assert_reasoning_matches_parse(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn gemma4_answers_without_reasoning_when_thinking_is_disabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, SINGLE_WORD_INSTRUCTION, "[]", false)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut LlamaSampler::greedy()?,
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    answer.assert_content_without_reasoning(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn gemma4_reasoning_streams_match_the_parsed_answer_when_reasoning_is_prefilled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            SINGLE_WORD_INSTRUCTION.to_owned(),
        )?],
        "[]",
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::ContinueReasoning(
                "The user wants one word.".to_owned(),
            ),
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut LlamaSampler::greedy()?,
        max_generated_tokens: MAX_REASONING_ANSWER_TOKENS,
    }
    .run();

    answer.assert_reasoning_matches_parse(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn glm47_answers_without_reasoning_when_thinking_is_disabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", false)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    answer.assert_content_without_reasoning(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn glm47_reasoning_streams_match_the_parsed_answer_when_thinking_is_enabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", true)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_REASONING_ANSWER_TOKENS,
    }
    .run();

    answer.assert_reasoning_matches_parse(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn mistral3_answers_without_reasoning_after_its_reasoning_is_closed(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = closed_reasoning_chat_session(fixture.model, SINGLE_WORD_INSTRUCTION, "[]");
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut LlamaSampler::greedy()?,
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    answer.assert_content_without_reasoning(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn mistral3_reasoning_streams_match_the_parsed_answer_when_thinking_is_enabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, SINGLE_WORD_INSTRUCTION, "[]", true)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut LlamaSampler::greedy()?,
        max_generated_tokens: MAX_REASONING_ANSWER_TOKENS,
    }
    .run();

    answer.assert_reasoning_matches_parse(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen35_answers_without_reasoning_when_thinking_is_disabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", false)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    answer.assert_content_without_reasoning(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen35_reasoning_streams_match_the_parsed_answer_when_thinking_is_enabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", true)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_REASONING_ANSWER_TOKENS,
    }
    .run();

    answer.assert_reasoning_matches_parse(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen36_answers_without_reasoning_when_thinking_is_disabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", false)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    answer.assert_content_without_reasoning(&session);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn qwen36_reasoning_streams_match_the_parsed_answer_when_thinking_is_enabled(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, ARITHMETIC_QUESTION, "[]", true)?;
    let answer = ChatAnswerGeneration {
        fixture,
        session: &session,
        sampler: &mut penalized_nucleus_sampler(fixture.model),
        max_generated_tokens: MAX_REASONING_ANSWER_TOKENS,
    }
    .run();

    answer.assert_reasoning_matches_parse(&session);

    Ok(())
}
