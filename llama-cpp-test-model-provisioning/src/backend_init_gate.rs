use std::sync::Mutex;

/// Serializes the tests of one process that initialize the llama backend, which llama.cpp allows
/// only once at a time.
pub static BACKEND_INIT_GATE: Mutex<()> = Mutex::new(());
