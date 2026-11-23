# llama-cpp-rs

Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp). Minimal, close to raw bindings, kept up to date with upstream.

## Quick start

```bash
git clone --recursive https://github.com/intentee/llama-cpp-bindings
cd llama-cpp-bindings
cargo build --release
```

Add `--features cuda` for GPU acceleration.

## Crates

The API is split per llama.cpp handle family. Each crate has one concern and one error enum.

| Crate | Description | Error enum |
|---|---|---|
| `llama-cpp-backend` | Process-global runtime: initialization, NUMA, devices, dynamic backends, log routing. Defines the device features (`cuda`, `metal`, `vulkan`, ...) | `BackendError` |
| `llama-cpp-params` | Model params, context params, fitting both to device memory | `ParamsError` |
| `llama-cpp-model` | A loaded model: metadata, vocabulary, tokenization, chat templates, decision head, LoRA adapters | `ModelError` |
| `llama-cpp-context` | Running a model: batches, decode and encode, logits and embeddings | `ContextError` |
| `llama-cpp-context-memory` | Editing the KV memory of a context | `ContextMemoryError` |
| `llama-cpp-context-state` | Saving and restoring context state | `ContextStateError` |
| `llama-cpp-sampler` | Sampler chains and grammar-constrained sampling (GBNF, llguidance) | `SamplerError` |
| `llama-cpp-mtmd` | Multimodal projector: bitmaps, chunks, encoding and evaluation | `MtmdError` |
| `llama-cpp-chat` | Chat sessions rendered by the model's chat template and parsing answers back into messages | `ChatError` |
| `llama-cpp-bindings` | The generation loop: classifying sampled tokens into reasoning, content and tool calls, and counting usage | `LlamaCppError` |
| `llama-cpp-gbnf` | GBNF grammars: derived from JSON schemas, validated by llama.cpp's parser | `GbnfError` |
| `llama-cpp-gguf` | Reading and rewriting GGUF files | `GgufContextError` |
| `llama-cpp-token-usage` | Token accounting | `TokenUsageError` |
| `llama-cpp-bindings-types` | Chat message value types, free of FFI | `ModelProvidedToolCallIdError` |
| `llama-cpp-path-c-string` | Converting a path into the C string llama.cpp takes | `PathCStringError` |
| `llama-cpp-error-chain` | Rendering an error with its source chain as one line | |
| `llama-cpp-error-recorder` | Capturing an error raised inside an FFI callback for the Rust code that drove the call | |
| `llama-cpp-log-decoder` | Decoding the llama.cpp / ggml log callback stream | |
| `llama-cpp-bindings-sys` | Raw FFI bindings generated against llama.cpp, the C++ wrapper they call, and the owner of strings the C++ side allocates | |
| `llama-cpp-native` | llama.cpp built from its sources with CMake, linked as the native library the bindings call into | |
| `llama-cpp-bindings-build` | Build scripts of `llama-cpp-native` and `llama-cpp-bindings-sys` | `BuildError` |

Every item is imported from the module named after it, for example:

```rust
use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_chat::chat_session::ChatSession;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
```

Device features are selected on `llama-cpp-bindings` (or directly on `llama-cpp-backend`), for example `llama-cpp-bindings = { version = "0.29", features = ["cuda"] }`.

## Decision models

llama.cpp decision models (OpenJev, Laya, Clef, ...) score options instead of generating text.

- `LlamaModel::decision_type()` returns the model's `DecisionType` (`DecisionType::None` for a text-generation model), and `LlamaModel::decision_labels()` returns the labels its metadata declares.
- `GgufInfo::read(path)` (`llama_cpp_model::gguf_info::GgufInfo`) reads the decision type, the training context size and whether a chat template is present from a GGUF file without loading the model; its failures are `ModelError`.
- `LlamaModel::chat_template(Some(name))?.render(globals_json)` renders a named template with a JSON object of globals, which is how a decision prompt is built.
- `LlamaBatch::set_decision_order(token_index, decision_order)` sets the `DecisionOrder` of one batch entry (`None`, `QuestionNoul`, `QuestionChoice`, `QuestionScore`, `Option`).
- `LlamaModel::token_mask()` returns the vocabulary's mask token, and `LlamaModel::token_text_bytes(token)` the raw text of a token.

```rust
use llama_cpp_context::decision_order::DecisionOrder;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::decision_type::DecisionType;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;

assert_eq!(model.decision_type()?, DecisionType::Clef);

let prompt = model
    .chat_template(Some("systemone"))?
    .render(&globals.to_string())?;
let mut batch = LlamaBatch::new(context.limits());

for token in model.str_to_token(&prompt, AddBos::Never, ParseSpecialTokens::Always)? {
    let token_index = batch.n_tokens();

    batch.add(SampledToken::Content(token), token_index, &[0], true)?;
    batch.set_decision_order(token_index, DecisionOrder::Option)?;
}

context.decode(&batch)?;

let option_score = context.embeddings_ith(0)?.first().copied();
```

How scores are read depends on the decision type: OpenJev puts them in the logits of its label tokens, Laya in the embedding of each mask token, Clef in the embeddings of the entries marked with a decision order. The tests in `llama-cpp-bindings-tests/tests/decision_*.rs` show each.
