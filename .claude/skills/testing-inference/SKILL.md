---
name: testing-inference
description: Runs model inference to verify text generation works. Use after modifying token decoding, model loading, context creation, sampling, or any code path that affects text generation.
---

# Inference Testing

Run the `usage` example which auto-downloads a small model and generates a response:

```bash
cargo run --example usage
```

Model is `unsloth/Qwen3.5-0.8B-GGUF` (cached after first download). Confirm the output contains coherent text and exits without errors. Any panic, segfault, or garbled output indicates a regression.
