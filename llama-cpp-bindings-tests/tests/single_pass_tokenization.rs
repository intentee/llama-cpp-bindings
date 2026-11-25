use anyhow::Result;
use anyhow::ensure;

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
fn text_of_one_token_per_byte_is_tokenized_by_a_single_llama_cpp_call(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let digits = "0123456789".repeat(10);
    let _fault = ArmedFault::after_passing(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_TOKENIZE_THROWS,
        1,
    );

    let tokens = model.str_to_token(&digits, AddBos::Never, ParseSpecialTokens::Never)?;

    ensure!(tokens.len() == digits.len());

    Ok(())
}
