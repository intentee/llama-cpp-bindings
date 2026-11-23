use serial_test::serial;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_backend::llama_backend_device::LlamaBackendDevice;
use llama_cpp_backend::llama_backend_device_type::LlamaBackendDeviceType;

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
