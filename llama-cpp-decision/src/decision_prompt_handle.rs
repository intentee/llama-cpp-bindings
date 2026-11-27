use std::ptr::NonNull;

pub struct DecisionPromptHandle {
    pub handle: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt>,
}

unsafe impl Send for DecisionPromptHandle {}

unsafe impl Sync for DecisionPromptHandle {}

impl Drop for DecisionPromptHandle {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_prompt_free(self.handle.as_ptr());
        };
    }
}
