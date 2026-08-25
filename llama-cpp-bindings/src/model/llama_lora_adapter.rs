use std::marker::PhantomData;
use std::ptr::NonNull;

use super::LlamaModel;
unsafe fn free_lora_adapter(adapter: *mut llama_cpp_bindings_sys::llama_adapter_lora) {
    unsafe { llama_cpp_bindings_sys::llama_adapter_lora_free(adapter) }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct LlamaLoraAdapter<'model> {
    lora_adapter: *mut llama_cpp_bindings_sys::llama_adapter_lora,
    model: PhantomData<&'model LlamaModel>,
}

impl LlamaLoraAdapter<'_> {
    pub(crate) const fn new(
        lora_adapter: NonNull<llama_cpp_bindings_sys::llama_adapter_lora>,
    ) -> Self {
        Self {
            lora_adapter: lora_adapter.as_ptr(),
            model: PhantomData,
        }
    }

    pub(crate) const fn as_ptr(&self) -> *mut llama_cpp_bindings_sys::llama_adapter_lora {
        self.lora_adapter
    }
}

impl Drop for LlamaLoraAdapter<'_> {
    fn drop(&mut self) {
        unsafe { free_lora_adapter(self.lora_adapter) }
    }
}

#[cfg(test)]
mod ownership_tests {
    use std::marker::PhantomData;
    use std::mem::ManuallyDrop;
    use std::ptr::NonNull;

    use super::LlamaLoraAdapter;

    #[test]
    fn adapter_preserves_the_owned_native_pointer() {
        let pointer = NonNull::dangling();
        let adapter = ManuallyDrop::new(LlamaLoraAdapter::new(pointer));

        assert_eq!(adapter.as_ptr(), pointer.as_ptr());
    }

    #[test]
    fn dropping_an_adapter_releases_its_native_pointer() {
        drop(LlamaLoraAdapter {
            lora_adapter: std::ptr::null_mut(),
            model: PhantomData,
        });
    }
}
