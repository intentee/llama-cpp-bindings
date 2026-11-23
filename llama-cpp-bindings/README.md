# llama-cpp-bindings

Safe Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp).

This crate holds the generation loop (`SampledTokenClassifier`) and forwards the device features. The rest of the API lives in the crates it builds on: `llama-cpp-backend`, `llama-cpp-params`, `llama-cpp-model`, `llama-cpp-context`, `llama-cpp-sampler`, `llama-cpp-mtmd` and `llama-cpp-chat`.
