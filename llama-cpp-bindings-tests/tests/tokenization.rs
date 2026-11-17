use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
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
fn the_first_token_of_tokenized_text_decodes_to_its_first_word(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let tokens = model.str_to_token("hello world", AddBos::Never, ParseSpecialTokens::Always)?;
    ensure!(tokens != Vec::<LlamaToken>::new());
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let piece = model.token_to_piece(
        SampledToken::Content(*tokens.first().context("tokens has no element 0")?),
        &mut decoder,
        false,
        None,
    )?;

    ensure!(piece == "hello");

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
fn str_to_token_prepends_bos_for_a_model_that_adds_it(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let tokens_without_bos =
        model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;
    let expected = model
        .token_bos()
        .into_iter()
        .chain(tokens_without_bos)
        .collect::<Vec<_>>();

    ensure!(model.adds_bos_token());
    ensure!(model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)? == expected);

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
fn str_to_token_adds_nothing_for_a_model_that_adds_no_bos(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(!model.adds_bos_token());
    ensure!(
        model.str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?
            == model.str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?
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
fn str_to_token_regrows_its_buffer_for_more_tokens_than_it_estimated(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let many_numbers = (0_u32..2000)
        .map(|number| number.to_string())
        .collect::<Vec<String>>()
        .join(" ");

    let tokens = model.str_to_token(&many_numbers, AddBos::Always, ParseSpecialTokens::Always)?;

    ensure!(tokens.len() > many_numbers.len() / 2);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn tokenizing_without_special_token_parsing_keeps_control_markup_as_text(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let markup = "<|fim_prefix|>";
    let parsed = model.str_to_token(markup, AddBos::Never, ParseSpecialTokens::Always)?;
    let plain = model.str_to_token(markup, AddBos::Never, ParseSpecialTokens::Never)?;

    ensure!(parsed.len() == 1);
    ensure!(plain.len() > 1);
    ensure!(!plain.contains(parsed.first().context("parsed has no element 0")?));

    Ok(())
}
