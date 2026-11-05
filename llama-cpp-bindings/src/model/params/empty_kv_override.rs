use std::mem;

/// A KV override with every byte zero, which llama.cpp reads as the end of an override list.
pub const EMPTY_KV_OVERRIDE: llama_cpp_bindings_sys::llama_model_kv_override =
    unsafe { mem::zeroed() };

/// How many bytes, its NUL included, a string KV override value holds in llama.cpp.
pub const KV_OVERRIDE_TEXT_CAPACITY: usize =
    unsafe { EMPTY_KV_OVERRIDE.__bindgen_anon_1.val_str }.len();
