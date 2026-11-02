# llama-cpp-bindings-sys

Low level FFI bindings to [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Private llama.cpp headers

The wrappers use llama.cpp's public headers, with two deliberate exceptions where llama.cpp exposes no public equivalent:

- `src/llama-model.h` (`wrapper_context.cpp`): reads the model's default causal attention before a context exists. Context creation aborts the process on a `GGML_ASSERT` when the context's sequences exceed its output capacity, and that capacity depends on whether attention is causal, so the bindings validate it first. `llama_get_causal_attn` only answers once a context exists.
- `src/llama-grammar.h` (`wrapper_gbnf.cpp`): validates GBNF without a model and reports why a grammar is rejected (syntax error, empty rule set, missing root symbol, unsupported grammar). `llama_sampler_init_grammar` only returns `NULL` and writes the reason to stderr.
