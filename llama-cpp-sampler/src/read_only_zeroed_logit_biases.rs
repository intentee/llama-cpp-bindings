use std::ffi::c_void;
use std::io;
use std::ops::Deref;
use std::ptr;
use std::slice;

use llama_cpp_model::llama_logit_bias::LlamaLogitBias;

pub struct ReadOnlyZeroedLogitBiases {
    mapping: *mut c_void,
    bias_count: usize,
}

impl ReadOnlyZeroedLogitBiases {
    pub fn new(bias_count: usize) -> Self {
        let mapping = unsafe {
            libc::mmap(
                ptr::null_mut(),
                Self::byte_len(bias_count),
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        assert_ne!(mapping, libc::MAP_FAILED, "{}", io::Error::last_os_error());

        Self {
            mapping,
            bias_count,
        }
    }

    fn byte_len(bias_count: usize) -> usize {
        bias_count
            .checked_mul(size_of::<LlamaLogitBias>())
            .expect("the byte length of the logit biases must fit a usize")
    }
}

impl Deref for ReadOnlyZeroedLogitBiases {
    type Target = [LlamaLogitBias];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.mapping.cast::<LlamaLogitBias>(), self.bias_count) }
    }
}

impl Drop for ReadOnlyZeroedLogitBiases {
    fn drop(&mut self) {
        let unmap_status = unsafe { libc::munmap(self.mapping, Self::byte_len(self.bias_count)) };

        assert_eq!(unmap_status, 0, "{}", io::Error::last_os_error());
    }
}
