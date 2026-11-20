use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const EVALUATED_IMAGE_DIMENSIONS: [ImageDimensions; 10] = [
    ImageDimensions {
        width: 224,
        height: 224,
    },
    ImageDimensions {
        width: 512,
        height: 512,
    },
    ImageDimensions {
        width: 100,
        height: 500,
    },
    ImageDimensions {
        width: 337,
        height: 421,
    },
    ImageDimensions {
        width: 2,
        height: 2,
    },
    ImageDimensions {
        width: 7,
        height: 13,
    },
    ImageDimensions {
        width: 3,
        height: 1000,
    },
    ImageDimensions {
        width: 1000,
        height: 3,
    },
    ImageDimensions {
        width: 1920,
        height: 1080,
    },
    ImageDimensions {
        width: 4096,
        height: 4096,
    },
];

fn eval_synthetic_bitmap(fixture: &LlamaFixture<'_>, width: u32, height: u32) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let image_data =
        vec![
            128u8;
            (width as usize)
                .checked_mul(height as usize)
                .context("`(width as usize) * (height as usize)` must not overflow")?
                .checked_mul(3)
                .context("`(width as usize) * (height as usize) * 3` must not overflow")?
        ];
    let bitmap = MtmdBitmap::from_image_data(width, height, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(
        n_positions
            .checked_add(256)
            .context("`n_positions + 256` must not overflow")?,
    )?;
    if fixture.context_params.n_ctx < required_n_ctx {
        bail!(
            "fixture n_ctx ({}) below required ({}) for {}x{} image",
            fixture.context_params.n_ctx,
            required_n_ctx,
            width,
            height,
        );
    }

    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, false)? == n_positions,
        "evaluating a {width}x{height} image must advance past every chunk position"
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
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_refuses_chunks_that_run_past_the_max_position_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;
    let second_chunk = chunks.get(1).context("the image follows the first chunk")?;
    let mut llama_ctx = fixture.build_context()?;
    let max_position = llama_ctx.limits().max_position();
    let start_position = max_position
        .checked_sub(first_chunk.n_positions())
        .context("`max_position - first_chunk.n_positions()` must not overflow")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(first_chunk.position_after(&llama_ctx, start_position) == Ok(max_position));
    ensure!(
        chunks.eval_chunks(&mut llama_ctx, start_position, 0, n_batch, true)
            == Err(MtmdError::ChunkRunsPastMaxPosition {
                start_position: max_position,
                n_tokens: i32::try_from(second_chunk.n_tokens())?,
                n_positions: second_chunk.n_positions(),
                max_position,
            })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_eval_advances_past_every_position_of_the_chunk(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(first_chunk.eval(&mut llama_ctx, 0, 0, n_batch, true) == Ok(first_chunk.n_positions()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_eval_refuses_a_chunk_that_runs_past_the_max_position_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        first_chunk.eval(&mut llama_ctx, i32::MAX, 0, n_batch, true)
            == Err(MtmdError::ChunkRunsPastMaxPosition {
                start_position: i32::MAX,
                n_tokens: i32::try_from(first_chunk.n_tokens())?,
                n_positions: first_chunk.n_positions(),
                max_position: llama_ctx.limits().max_position(),
            })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_advances_past_every_position_of_a_standard_image(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;

    let fixtures = fixtures_dir();
    let image_path = fixtures.join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;
    let input_text = MtmdInputText {
        text: "What is in this image? <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(
        n_positions
            .checked_add(256)
            .context("`n_positions + 256` must not overflow")?,
    )?;
    ensure!(
        fixture.context_params.n_ctx >= required_n_ctx,
        "fixture n_ctx ({}) below required ({}); update the attribute literal",
        fixture.context_params.n_ctx,
        required_n_ctx,
    );

    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, false) == Ok(n_positions));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_evaluates_images_of_ordinary_and_extreme_dimensions(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    for ImageDimensions { width, height } in EVALUATED_IMAGE_DIMENSIONS {
        eval_synthetic_bitmap(fixture, width, height)
            .context(format!("dimension {width}x{height} should succeed"))?;
    }

    Ok(())
}

struct ImageDimensions {
    width: u32,
    height: u32,
}
