use std::sync::Mutex;

#[cfg(test)]
pub static BACKEND_INIT_GATE: Mutex<()> = Mutex::new(());
