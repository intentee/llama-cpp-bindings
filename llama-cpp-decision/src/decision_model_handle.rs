use std::ptr::NonNull;
use std::sync::Arc;

use llama_cpp_model::llama_model::LlamaModel;

pub struct DecisionModelHandle {
    pub handle: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_decision_model>,
    pub llama_model: Arc<LlamaModel>,
}

unsafe impl Send for DecisionModelHandle {}

unsafe impl Sync for DecisionModelHandle {}

impl Drop for DecisionModelHandle {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_model_free(self.handle.as_ptr());
        };
    }
}
