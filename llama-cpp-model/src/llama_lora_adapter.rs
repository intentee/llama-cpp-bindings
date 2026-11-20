use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct LlamaLoraAdapter<'model> {
    lora_adapter: NonNull<llama_cpp_bindings_sys::bindings::llama_adapter_lora>,
    model: PhantomData<&'model llama_cpp_bindings_sys::bindings::llama_model>,
}

impl LlamaLoraAdapter<'_> {
    /// Borrows an adapter its model owns; llama.cpp frees the adapter together with that model.
    ///
    /// # Safety
    ///
    /// `lora_adapter` must come from `llama_adapter_lora_init` for the model this handle borrows,
    /// and must live as long as that model, so nothing may free it with `llama_adapter_lora_free`.
    #[must_use]
    pub const unsafe fn new(
        lora_adapter: NonNull<llama_cpp_bindings_sys::bindings::llama_adapter_lora>,
    ) -> Self {
        Self {
            lora_adapter,
            model: PhantomData,
        }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut llama_cpp_bindings_sys::bindings::llama_adapter_lora {
        self.lora_adapter.as_ptr()
    }
}
