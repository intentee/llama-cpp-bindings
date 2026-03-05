# llama-cpp-rs

Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp). Minimal, close to raw bindings, kept up to date with upstream.

## Quick start

```bash
git clone --recursive https://github.com/intentee/llama-cpp-bindings
cd llama-cpp-rs
cargo build --release
```

Add `--features cuda` for GPU acceleration.

## Crates

| Crate | Description |
|---|---|
| `llama-cpp-bindings` | Safe Rust API |
| `llama-cpp-bindings-sys` | Raw FFI bindings |
| `llama-cpp-bindings-build` | Build system |
