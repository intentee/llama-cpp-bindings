#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![expect(
    non_camel_case_types,
    reason = "bindgen emits C struct and enum names verbatim and they don't follow Rust naming"
)]
#![expect(
    clippy::pub_underscore_fields,
    clippy::use_self,
    reason = "bindgen writes this file from the llama.cpp headers; its shape is not ours to change"
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
