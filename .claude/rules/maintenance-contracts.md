# Maintenance Contracts That Tooling Cannot Enforce

Two facts about this repository cannot be checked automatically. Honour them by hand.

## Reconcile `DEPRECATED_FUNCTIONS` on every llama.cpp bump

`llama-cpp-bindings-build/src/bindgen_config.rs` keeps an explicit list of the upstream
functions marked `DEPRECATED(...)` / `GGML_DEPRECATED(...)`, so they stay out of the
generated FFI surface. bindgen cannot derive it: it never emits `#[deprecated]`, no
`ParseCallbacks` hook can see attributes, and unmatched *blocklist* patterns are reported
nowhere, so a stale entry is silent.

After changing the `llama.cpp` submodule, re-reconcile the list against
`include/llama.h`, `ggml/include/ggml.h` and `tools/mtmd/mtmd.h`, in both directions:
entries that no longer exist upstream, and newly deprecated functions that are missing.

Read the declarations, do not grep for them. Two forms defeat line-oriented matching:
`GGML_DEPRECATED(` can sit on the line above the identifier, and several llama.h entries
are written `LLAMA_API DEPRECATED(...)` with the export macro first.

## Bump every workspace version together

`Cargo.toml` states `0.13.0` in `[workspace.package]` and again in each path entry under
`[workspace.dependencies]`. Cargo has no interpolation for dependency versions, so a
release bump must change all of them in one edit. A partial bump stays invisible until
publish time.
