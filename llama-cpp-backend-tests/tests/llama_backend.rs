use serial_test::serial;

use llama_cpp_backend::backend_error::BackendError;
use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_backend::llama_backend_device::LlamaBackendDevice;
use llama_cpp_backend::llama_backend_device_type::LlamaBackendDeviceType;
use llama_cpp_backend::numa_strategy::NumaStrategy;

#[test]
#[serial]
fn initializing_the_backend_returns_it() {
    assert_eq!(LlamaBackend::init(), Ok(LlamaBackend {}));
}

#[test]
#[serial]
fn initializing_the_backend_twice_reports_it_is_already_initialized() {
    let _backend = LlamaBackend::init().unwrap();

    assert_eq!(LlamaBackend::init(), Err(BackendError::AlreadyInitialized));
}

#[test]
#[serial]
fn init_numa_reports_an_already_initialized_backend() {
    let _backend = LlamaBackend::init().unwrap();

    assert_eq!(
        LlamaBackend::init_numa(NumaStrategy::Disabled),
        Err(BackendError::AlreadyInitialized)
    );
}

#[test]
#[serial]
fn the_backend_supports_mmap_and_mlock() {
    let _backend = LlamaBackend::init().unwrap();

    assert_eq!(
        [
            LlamaBackend::supports_mmap(),
            LlamaBackend::supports_mlock()
        ],
        [true, true]
    );
}

#[test]
#[serial]
fn the_backend_offloads_to_a_gpu_exactly_when_one_is_registered() {
    let _backend = LlamaBackend::init().unwrap();
    let offloads_to_a_gpu = LlamaBackend::supports_gpu_offload();

    assert_eq!(
        offloads_to_a_gpu,
        LlamaBackendDevice::list_all()
            .unwrap()
            .iter()
            .any(|device| matches!(
                device.device_type,
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
            ))
    );
}

#[test]
#[serial]
fn dropping_the_backend_allows_reinitializing_it() {
    let backend = LlamaBackend::init().unwrap();
    drop(backend);

    assert_eq!(LlamaBackend::init(), Ok(LlamaBackend {}));
}

#[test]
#[serial]
fn init_numa_initializes_the_backend_with_a_numa_strategy() {
    assert_eq!(
        LlamaBackend::init_numa(NumaStrategy::Distribute),
        Ok(LlamaBackend {})
    );
}
