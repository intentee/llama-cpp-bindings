use anyhow::Result;

use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn the_first_token_of_tokenized_text_decodes_to_its_first_word(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let tokens = model.str_to_token("hello world", AddBos::Never, ParseSpecialTokens::Always)?;
    assert_ne!(tokens, Vec::<LlamaToken>::new());
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let piece =
        model.token_to_piece(&SampledToken::Content(tokens[0]), &mut decoder, false, None)?;

    assert_eq!(piece, "hello");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn str_to_token_prepends_bos_for_a_model_that_adds_it(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let tokens_without_bos =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;
    let expected = model
        .token_bos()
        .into_iter()
        .chain(tokens_without_bos)
        .collect::<Vec<_>>();

    assert!(model.adds_bos_token());
    assert_eq!(
        model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?,
        expected
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn str_to_token_adds_nothing_for_a_model_that_adds_no_bos(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;

    assert!(!model.adds_bos_token());
    assert_eq!(
        model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?,
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn str_to_token_regrows_its_buffer_for_more_tokens_than_it_estimated(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let many_numbers = (0_u32..2000)
        .map(|number| number.to_string())
        .collect::<Vec<String>>()
        .join(" ");

    let tokens =
        fixture
            .model
            .str_to_token(&many_numbers, AddBos::Always, ParseSpecialTokens::Always)?;

    assert!(tokens.len() > many_numbers.len() / 2);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn tokenizing_without_special_token_parsing_keeps_control_markup_as_text(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let markup = "<|fim_prefix|>";
    let parsed = fixture
        .model
        .str_to_token(markup, AddBos::Never, ParseSpecialTokens::Always)?;
    let plain = fixture
        .model
        .str_to_token(markup, AddBos::Never, ParseSpecialTokens::Never)?;

    assert_eq!(parsed.len(), 1);
    assert!(plain.len() > 1);
    assert!(!plain.contains(&parsed[0]));

    Ok(())
}
