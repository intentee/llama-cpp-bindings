use enumflags2::bitflags;
use llama_cpp_bindings_sys::bindings::LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
use llama_cpp_bindings_sys::bindings::LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
pub enum LlamaStateSeqFlag {
    PartialOnly = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,
    OnDevice = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
}
