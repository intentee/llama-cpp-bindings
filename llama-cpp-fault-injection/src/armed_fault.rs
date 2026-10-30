use std::marker::PhantomData;

/// Makes one C++ wrapper fault point fail on the current thread until the guard is dropped.
///
/// The fault point is armed per thread, so the guard cannot leave the thread that armed it.
#[derive(Debug)]
pub struct ArmedFault {
    armed_on_this_thread: PhantomData<*const ()>,
}

impl ArmedFault {
    /// Fails every time the wrappers reach `point`.
    #[must_use]
    pub fn at(point: llama_cpp_bindings_sys::llama_rs_fault_point) -> Self {
        Self::after_passing(point, 0)
    }

    /// Lets the wrappers pass `point` `passing_hits` times, then fails every later time.
    #[must_use]
    pub fn after_passing(
        point: llama_cpp_bindings_sys::llama_rs_fault_point,
        passing_hits: usize,
    ) -> Self {
        unsafe { llama_cpp_bindings_sys::llama_rs_fault_injection_arm(point, passing_hits) };

        Self {
            armed_on_this_thread: PhantomData,
        }
    }
}

impl Drop for ArmedFault {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_rs_fault_injection_disarm() };
    }
}
