use std::ptr::NonNull;

pub struct DecisionRequestHandle {
    pub handle: NonNull<llama_cpp_bindings_sys::bindings::llama_rs_decision_request>,
}

unsafe impl Send for DecisionRequestHandle {}

unsafe impl Sync for DecisionRequestHandle {}

impl Drop for DecisionRequestHandle {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_decision_request_free(self.handle.as_ptr());
        };
    }
}
