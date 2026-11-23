use std::ffi::{CStr, CString};
use std::fs;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};
use std::ptr;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_path_c_string::path_c_string::path_c_string;

use crate::gguf_context_error::GgufContextError;
use crate::gguf_tensor_f32::GgufTensorF32;
use crate::gguf_type::GgufType;

const F32_SIZE_IN_BYTES: usize = size_of::<f32>();

#[derive(Debug)]
pub struct GgufContext {
    context: NonNull<llama_cpp_bindings_sys::bindings::gguf_context>,
    path: PathBuf,
}

impl GgufContext {
    /// # Errors
    ///
    /// Returns [`GgufContextError::InitFailed`] if the file cannot be opened or parsed.
    /// Returns [`GgufContextError::LoadPathInvalid`] if the path is not valid UTF-8 or contains a
    /// null byte.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, GgufContextError> {
        let path_ref = path.as_ref();
        let c_path = path_c_string(path_ref).map_err(GgufContextError::LoadPathInvalid)?;

        let init_params = llama_cpp_bindings_sys::bindings::gguf_init_params {
            no_alloc: true,
            ctx: ptr::null_mut(),
        };

        let raw = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_init_from_file(c_path.as_ptr(), init_params)
        };
        let context = NonNull::new(raw)
            .ok_or_else(|| GgufContextError::InitFailed(path_ref.to_path_buf()))?;

        Ok(Self {
            context,
            path: path_ref.to_path_buf(),
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyNotFound`] if the key does not exist.
    /// Returns [`GgufContextError::KeyContainsNulByte`] if the key contains a null byte.
    pub fn find_key(&self, key: &str) -> Result<i64, GgufContextError> {
        let c_key = CString::new(key).map_err(GgufContextError::KeyContainsNulByte)?;
        let index = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_find_key(self.context.as_ptr(), c_key.as_ptr())
        };

        if index < 0 {
            return Err(GgufContextError::KeyNotFound {
                key: key.to_string(),
            });
        }

        Ok(index)
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] if the file holds no key with this id.
    /// Returns [`GgufContextError::KeyNameNotUtf8`] if the key name is not valid UTF-8.
    pub fn key_at(&self, key_id: i64) -> Result<&str, GgufContextError> {
        self.require_key_id_in_range(key_id)?;

        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::bindings::gguf_get_key(
                self.context.as_ptr(),
                key_id,
            ))
        };

        c_str
            .to_str()
            .map_err(|source| GgufContextError::KeyNameNotUtf8 { key_id, source })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] if the file holds no key with this id.
    /// Returns [`GgufContextError::UnknownValueType`] if the value type is not a known GGUF type.
    pub fn kv_type(&self, key_id: i64) -> Result<GgufType, GgufContextError> {
        self.require_key_id_in_range(key_id)?;

        let raw_type = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_kv_type(self.context.as_ptr(), key_id)
        };

        GgufType::from_raw(raw_type).ok_or(GgufContextError::UnknownValueType { key_id, raw_type })
    }

    #[must_use]
    pub fn n_kv(&self) -> i64 {
        unsafe { llama_cpp_bindings_sys::bindings::gguf_get_n_kv(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn n_tensors(&self) -> i64 {
        unsafe { llama_cpp_bindings_sys::bindings::gguf_get_n_tensors(self.context.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::TensorNotFound`] if the file holds no tensor with this name.
    /// Returns [`GgufContextError::TensorIsNotF32`] if the tensor is stored in another type.
    /// Returns [`GgufContextError::TensorReadFailed`] if the tensor data cannot be read.
    /// Returns [`GgufContextError::TensorNameContainsNulByte`] if the name contains a null byte.
    pub fn read_tensor_f32(&self, name: &str) -> Result<GgufTensorF32, GgufContextError> {
        let c_name = CString::new(name).map_err(GgufContextError::TensorNameContainsNulByte)?;
        let tensor_id = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_find_tensor(
                self.context.as_ptr(),
                c_name.as_ptr(),
            )
        };

        if tensor_id < 0 {
            return Err(GgufContextError::TensorNotFound {
                name: name.to_owned(),
            });
        }

        let ggml_type = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_tensor_type(self.context.as_ptr(), tensor_id)
        };

        if ggml_type != llama_cpp_bindings_sys::bindings::GGML_TYPE_F32 {
            return Err(GgufContextError::TensorIsNotF32 {
                name: name.to_owned(),
                ggml_type,
            });
        }

        Ok(GgufTensorF32 {
            shape: self.tensor_shape(tensor_id),
            values: self.tensor_f32_values(name, tensor_id)?,
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyNotFound`] if the key does not exist.
    /// Returns [`GgufContextError::KeyContainsNulByte`] if the key contains a null byte.
    pub fn remove_key(&mut self, key: &str) -> Result<(), GgufContextError> {
        let key_id = self.find_key(key)?;
        let stored_key = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_key(self.context.as_ptr(), key_id)
        };

        unsafe {
            llama_cpp_bindings_sys::bindings::gguf_remove_key(self.context.as_ptr(), stored_key)
        };

        Ok(())
    }

    pub fn set_arr_i32(&mut self, key: &CStr, values: &[i32]) {
        unsafe {
            llama_cpp_bindings_sys::bindings::gguf_set_arr_data(
                self.context.as_ptr(),
                key.as_ptr(),
                GgufType::Int32.to_raw(),
                values.as_ptr().cast(),
                values.len(),
            );
        }
    }

    pub fn set_val_str(&mut self, key: &CStr, value: &CStr) {
        unsafe {
            llama_cpp_bindings_sys::bindings::gguf_set_val_str(
                self.context.as_ptr(),
                key.as_ptr(),
                value.as_ptr(),
            );
        }
    }

    pub fn set_val_u32(&mut self, key: &CStr, value: u32) {
        unsafe {
            llama_cpp_bindings_sys::bindings::gguf_set_val_u32(
                self.context.as_ptr(),
                key.as_ptr(),
                value,
            );
        }
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Array`] value.
    /// Returns [`GgufContextError::ArrayElementTypeMismatch`] if the array does not hold
    /// [`GgufType::Int32`] elements.
    pub fn val_arr_i32(&self, key_id: i64) -> Result<&[i32], GgufContextError> {
        self.require_value_type(key_id, GgufType::Array)?;

        let element_type = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_arr_type(self.context.as_ptr(), key_id)
        };

        if element_type != GgufType::Int32.to_raw() {
            return Err(GgufContextError::ArrayElementTypeMismatch {
                key_id,
                expected: GgufType::Int32,
                actual_raw_type: element_type,
            });
        }

        Ok(unsafe {
            slice::from_raw_parts(
                llama_cpp_bindings_sys::bindings::gguf_get_arr_data(self.context.as_ptr(), key_id)
                    .cast::<i32>(),
                llama_cpp_bindings_sys::bindings::gguf_get_arr_n(self.context.as_ptr(), key_id),
            )
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Float32`] value.
    pub fn val_f32(&self, key_id: i64) -> Result<f32, GgufContextError> {
        self.require_value_type(key_id, GgufType::Float32)?;

        Ok(unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_val_f32(self.context.as_ptr(), key_id)
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Int32`] value.
    pub fn val_i32(&self, key_id: i64) -> Result<i32, GgufContextError> {
        self.require_value_type(key_id, GgufType::Int32)?;

        Ok(unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_val_i32(self.context.as_ptr(), key_id)
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::String`] value.
    /// Returns [`GgufContextError::StringValueNotUtf8`] if the string value is not valid UTF-8.
    pub fn val_str(&self, key_id: i64) -> Result<&str, GgufContextError> {
        self.require_value_type(key_id, GgufType::String)?;

        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::bindings::gguf_get_val_str(
                self.context.as_ptr(),
                key_id,
            ))
        };

        c_str
            .to_str()
            .map_err(|source| GgufContextError::StringValueNotUtf8 { key_id, source })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Uint32`] value.
    pub fn val_u32(&self, key_id: i64) -> Result<u32, GgufContextError> {
        self.require_value_type(key_id, GgufType::Uint32)?;

        Ok(unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_val_u32(self.context.as_ptr(), key_id)
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Uint64`] value.
    pub fn val_u64(&self, key_id: i64) -> Result<u64, GgufContextError> {
        self.require_value_type(key_id, GgufType::Uint64)?;

        Ok(unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_val_u64(self.context.as_ptr(), key_id)
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::SourceFileReadFailed`] if the loaded file cannot be read.
    /// Returns [`GgufContextError::SourceFileTruncated`] if the loaded file ends before its
    /// tensor data begins.
    /// Returns [`GgufContextError::DestinationFileWriteFailed`] if the destination cannot be
    /// written.
    pub fn write_to_file(&self, destination: impl AsRef<Path>) -> Result<(), GgufContextError> {
        let destination_ref = destination.as_ref();
        let mut gguf_bytes =
            fs::read(&self.path).map_err(|source| GgufContextError::SourceFileReadFailed {
                path: self.path.clone(),
                source,
            })?;
        let data_offset = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_data_offset(self.context.as_ptr())
        };

        if gguf_bytes.len() < data_offset {
            return Err(GgufContextError::SourceFileTruncated {
                path: self.path.clone(),
                data_offset,
                file_size: gguf_bytes.len(),
            });
        }

        gguf_bytes.splice(..data_offset, self.metadata_bytes());

        fs::write(destination_ref, gguf_bytes).map_err(|source| {
            GgufContextError::DestinationFileWriteFailed {
                path: destination_ref.to_path_buf(),
                source,
            }
        })
    }

    fn metadata_bytes(&self) -> Vec<u8> {
        let mut metadata = vec![
            0;
            unsafe {
                llama_cpp_bindings_sys::bindings::gguf_get_meta_size(self.context.as_ptr())
            }
        ];

        unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_meta_data(
                self.context.as_ptr(),
                metadata.as_mut_ptr().cast(),
            );
        };

        metadata
    }

    fn require_key_id_in_range(&self, key_id: i64) -> Result<(), GgufContextError> {
        let n_kv = self.n_kv();

        if (0..n_kv).contains(&key_id) {
            Ok(())
        } else {
            Err(GgufContextError::KeyIdOutOfRange { key_id, n_kv })
        }
    }

    fn require_value_type(&self, key_id: i64, expected: GgufType) -> Result<(), GgufContextError> {
        let actual = self.kv_type(key_id)?;

        if actual == expected {
            Ok(())
        } else {
            Err(GgufContextError::ValueTypeMismatch {
                key_id,
                expected,
                actual,
            })
        }
    }

    fn tensor_f32_values(&self, name: &str, tensor_id: i64) -> Result<Vec<f32>, GgufContextError> {
        let data_offset = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_data_offset(self.context.as_ptr())
        };
        let tensor_offset = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_tensor_offset(
                self.context.as_ptr(),
                tensor_id,
            )
        };
        let tensor_size = unsafe {
            llama_cpp_bindings_sys::bindings::gguf_get_tensor_size(self.context.as_ptr(), tensor_id)
        };
        let read_failed = |source: io::Error| GgufContextError::TensorReadFailed {
            name: name.to_owned(),
            path: self.path.clone(),
            source,
        };

        let file = File::open(&self.path).map_err(read_failed)?;
        let mut bytes = vec![0; tensor_size];

        file.read_exact_at(&mut bytes, data_offset.saturating_add(tensor_offset) as u64)
            .map_err(read_failed)?;

        let (value_byte_chunks, _bytes_after_the_last_value) =
            bytes.as_chunks::<F32_SIZE_IN_BYTES>();

        Ok(value_byte_chunks
            .iter()
            .copied()
            .map(f32::from_ne_bytes)
            .collect())
    }

    fn tensor_shape(
        &self,
        tensor_id: i64,
    ) -> [i64; llama_cpp_bindings_sys::bindings::GGML_MAX_DIMS as usize] {
        let element_counts = unsafe {
            slice::from_raw_parts(
                llama_cpp_bindings_sys::bindings::gguf_get_tensor_ne(
                    self.context.as_ptr(),
                    tensor_id,
                ),
                llama_cpp_bindings_sys::bindings::GGML_MAX_DIMS as usize,
            )
        };
        let mut shape = [0; llama_cpp_bindings_sys::bindings::GGML_MAX_DIMS as usize];

        shape.copy_from_slice(element_counts);

        shape
    }
}

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::bindings::gguf_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::ffi::OsStr;
    use std::fs;
    use std::fs::File;
    use std::io::Write as _;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process;

    use llama_cpp_error_chain::error_chain::error_chain;

    use super::GgufContext;
    use crate::gguf_type::GgufType;

    const GGUF_ALIGNMENT: usize = 32;
    const GGML_TYPE_F16: u32 = 1;

    fn vocabulary_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("ggml-vocab-bert-bge.gguf")
    }

    fn i32_array_bytes(values: &[i32]) -> Vec<u8> {
        let mut encoded = GgufType::Int32.to_raw().to_le_bytes().to_vec();

        encoded.extend_from_slice(&(values.len() as u64).to_le_bytes());

        for value in values {
            encoded.extend_from_slice(&value.to_le_bytes());
        }

        encoded
    }

    fn rewritten_gguf_path(test_name: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "llama_cpp_bindings_rewritten_{}_{}.gguf",
            process::id(),
            test_name,
        ))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn bias_and_weight_tensors(test_name: &str) -> SyntheticGgufFile {
        SyntheticGgufBuilder::default()
            .tensor(b"synthetic.bias", &[2], GGML_TYPE_F16, vec![0; 4])
            .tensor(
                b"synthetic.weight",
                &[3, 2],
                llama_cpp_bindings_sys::bindings::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )
            .write(test_name)
    }

    struct SyntheticTensor {
        name: Vec<u8>,
        shape: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }

    #[derive(Default)]
    struct SyntheticGgufBuilder {
        key_values: Vec<u8>,
        key_value_count: u64,
        tensors: Vec<SyntheticTensor>,
    }

    impl SyntheticGgufBuilder {
        fn string_value(self, key: &[u8], value: &[u8]) -> Self {
            let mut encoded = (value.len() as u64).to_le_bytes().to_vec();
            encoded.extend_from_slice(value);
            self.value(key, GgufType::String, &encoded)
        }

        fn tensor(mut self, name: &[u8], shape: &[u64], ggml_type: u32, data: Vec<u8>) -> Self {
            self.tensors.push(SyntheticTensor {
                name: name.to_vec(),
                shape: shape.to_vec(),
                ggml_type,
                data,
            });
            self
        }

        fn value(mut self, key: &[u8], gguf_type: GgufType, value: &[u8]) -> Self {
            self.key_values
                .extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.key_values.extend_from_slice(key);
            self.key_values
                .extend_from_slice(&gguf_type.to_raw().to_le_bytes());
            self.key_values.extend_from_slice(value);
            self.key_value_count = self.key_value_count.checked_add(1).unwrap();
            self
        }

        fn write(self, test_name: &str) -> SyntheticGgufFile {
            let mut bytes: Vec<u8> = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&self.key_value_count.to_le_bytes());
            bytes.extend_from_slice(&self.key_values);

            let mut data_section: Vec<u8> = Vec::new();

            for tensor in &self.tensors {
                bytes.extend_from_slice(&(tensor.name.len() as u64).to_le_bytes());
                bytes.extend_from_slice(&tensor.name);
                bytes.extend_from_slice(&u32::try_from(tensor.shape.len()).unwrap().to_le_bytes());
                for dimension in &tensor.shape {
                    bytes.extend_from_slice(&dimension.to_le_bytes());
                }
                bytes.extend_from_slice(&tensor.ggml_type.to_le_bytes());
                bytes.extend_from_slice(&(data_section.len() as u64).to_le_bytes());
                data_section.extend_from_slice(&tensor.data);
                data_section.resize(data_section.len().next_multiple_of(GGUF_ALIGNMENT), 0);
            }

            if !self.tensors.is_empty() {
                bytes.resize(bytes.len().next_multiple_of(GGUF_ALIGNMENT), 0);
                bytes.extend_from_slice(&data_section);
            }

            SyntheticGgufFile::from_bytes(test_name, &bytes)
        }
    }

    struct SyntheticGgufFile {
        path: PathBuf,
    }

    impl SyntheticGgufFile {
        fn from_bytes(test_name: &str, bytes: &[u8]) -> Self {
            let path = env::temp_dir().join(format!(
                "llama_cpp_bindings_synthetic_{}_{}.gguf",
                process::id(),
                test_name,
            ));

            let mut file = File::create(&path).unwrap();
            file.write_all(bytes).unwrap();

            Self { path }
        }
    }

    impl Drop for SyntheticGgufFile {
        fn drop(&mut self) {
            fs::remove_file(&self.path).unwrap();
        }
    }

    #[test]
    fn from_file_nonexistent_returns_init_failed() {
        let err = GgufContext::from_file("/nonexistent/file.gguf").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "Failed to initialize GGUF context from file: /nonexistent/file.gguf"
        );
    }

    #[test]
    fn n_kv_counts_every_key_of_the_file() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();

        assert_eq!(context.n_kv(), 20);
    }

    #[test]
    fn a_key_the_file_lacks_is_reported_not_found() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.find_key("nonexistent.key").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "Key not found in GGUF context: nonexistent.key"
        );
    }

    #[test]
    fn key_at_reads_the_name_of_the_key_find_key_located() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let key_name = context.key_at(index).unwrap();

        assert_eq!(key_name, "general.architecture");
    }

    #[test]
    fn key_at_rejects_a_key_id_past_the_last_key() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.key_at(context.n_kv()).unwrap_err();

        assert_eq!(
            error_chain(&err),
            "GGUF key id 20 is outside of the 20 keys the file holds"
        );
    }

    #[test]
    fn typed_value_rejects_a_negative_key_id() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.val_u32(-1).unwrap_err();

        assert_eq!(
            error_chain(&err),
            "GGUF key id -1 is outside of the 20 keys the file holds"
        );
    }

    #[test]
    fn kv_type_reports_a_string_key_as_string() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();

        assert_eq!(context.kv_type(index).unwrap(), GgufType::String);
    }

    #[test]
    fn val_str_reads_the_architecture_stored_in_the_file() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let value = context.val_str(index).unwrap();

        assert_eq!(value, "bert");
    }

    #[test]
    fn opening_a_path_that_is_not_utf8_is_refused() {
        let non_utf8_path = Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
        let err = GgufContext::from_file(non_utf8_path).unwrap_err();

        assert_eq!(
            error_chain(&err),
            "the GGUF file path cannot be handed to ggml: the path /tmp/\u{FFFD}\u{FFFD}.gguf is not valid UTF-8"
        );
    }

    #[test]
    fn opening_a_path_with_a_nul_byte_is_refused() {
        let err = GgufContext::from_file("/tmp/foo\0bar.gguf").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "the GGUF file path cannot be handed to ggml: the path contains a NUL byte: nul byte found in provided data at position: 8"
        );
    }

    #[test]
    fn finding_a_key_with_a_nul_byte_is_refused() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.find_key("foo\0bar").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "the GGUF key to look up contains a NUL byte: nul byte found in provided data at position: 3"
        );
    }

    #[test]
    fn typed_getters_read_the_values_a_gguf_file_stores() {
        let fixture = SyntheticGgufBuilder::default()
            .value(
                b"synthetic.i32_value",
                GgufType::Int32,
                &(-12345i32).to_le_bytes(),
            )
            .value(
                b"synthetic.u32_value",
                GgufType::Uint32,
                &4096u32.to_le_bytes(),
            )
            .value(
                b"synthetic.u64_value",
                GgufType::Uint64,
                &987_654_321u64.to_le_bytes(),
            )
            .value(
                b"synthetic.f32_value",
                GgufType::Float32,
                &2.41f32.to_le_bytes(),
            )
            .write("typed_values_round_trip");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let i32_index = context.find_key("synthetic.i32_value").unwrap();
        let u32_index = context.find_key("synthetic.u32_value").unwrap();
        let u64_index = context.find_key("synthetic.u64_value").unwrap();
        let f32_index = context.find_key("synthetic.f32_value").unwrap();

        assert_eq!(context.val_i32(i32_index).unwrap(), -12345);
        assert_eq!(context.val_u32(u32_index).unwrap(), 4096);
        assert_eq!(context.val_u64(u64_index).unwrap(), 987_654_321);
        assert!((context.val_f32(f32_index).unwrap() - 2.41).abs() < f32::EPSILON);
    }

    #[test]
    fn typed_values_reject_a_key_holding_another_type() {
        let fixture = SyntheticGgufBuilder::default()
            .value(b"synthetic.bool_value", GgufType::Bool, &[1])
            .write("typed_values_reject_another_type");
        let context = GgufContext::from_file(&fixture.path).unwrap();
        let bool_index = context.find_key("synthetic.bool_value").unwrap();
        let mismatches = [
            context.val_u32(bool_index).unwrap_err(),
            context.val_i32(bool_index).unwrap_err(),
            context.val_u64(bool_index).unwrap_err(),
            context.val_f32(bool_index).unwrap_err(),
            context.val_str(bool_index).unwrap_err(),
        ];

        assert_eq!(
            mismatches.map(|mismatch| error_chain(&mismatch)),
            ["Uint32", "Int32", "Uint64", "Float32", "String"].map(|expected| format!(
                "GGUF key id {bool_index} holds a Bool value, not {expected}"
            ))
        );
    }

    #[test]
    fn val_str_returns_utf8_error_for_non_utf8_value() {
        let fixture = SyntheticGgufBuilder::default()
            .string_value(b"synthetic.str_value", &[0xFF, 0xFE])
            .write("val_str_returns_utf8_error_for_non_utf8_value");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let value_index = context.find_key("synthetic.str_value").unwrap();
        let err = context.val_str(value_index).unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!(
                "GGUF key id {value_index} holds a string value that is not valid UTF-8: invalid utf-8 sequence of 1 bytes from index 0"
            )
        );
    }

    #[test]
    fn key_at_returns_utf8_error_for_non_utf8_key() {
        let fixture = SyntheticGgufBuilder::default()
            .value(&[0xFF, 0xFE], GgufType::Int32, &42i32.to_le_bytes())
            .write("key_at_returns_utf8_error_for_non_utf8_key");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let err = context.key_at(0).unwrap_err();

        assert_eq!(
            error_chain(&err),
            "GGUF key id 0 has a name that is not valid UTF-8: invalid utf-8 sequence of 1 bytes from index 0"
        );
    }

    #[test]
    fn n_tensors_counts_every_tensor_of_the_file() {
        let fixture = bias_and_weight_tensors("n_tensors_counts_every_tensor_of_the_file");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        assert_eq!(context.n_tensors(), 2);
    }

    #[test]
    fn read_tensor_f32_returns_shape_and_values() {
        let fixture = bias_and_weight_tensors("read_tensor_f32_returns_shape_and_values");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let tensor = context.read_tensor_f32("synthetic.weight").unwrap();

        assert_eq!(tensor.shape, [3, 2, 1, 1]);
        assert_eq!(tensor.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn read_tensor_f32_rejects_a_missing_tensor() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.read_tensor_f32("synthetic.missing").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "Tensor not found in GGUF context: synthetic.missing"
        );
    }

    #[test]
    fn read_tensor_f32_rejects_a_tensor_stored_in_another_type() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(b"synthetic.half", &[2], GGML_TYPE_F16, vec![0; 4])
            .write("read_tensor_f32_rejects_another_type");
        let context = GgufContext::from_file(&fixture.path).unwrap();
        let err = context.read_tensor_f32("synthetic.half").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "GGUF tensor synthetic.half holds ggml type 1, not F32"
        );
    }

    #[test]
    fn read_tensor_f32_reports_a_file_truncated_after_opening() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(
                b"synthetic.weight",
                &[2],
                llama_cpp_bindings_sys::bindings::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0]),
            )
            .write("read_tensor_f32_reports_a_file_truncated_after_opening");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        fs::write(&fixture.path, b"").unwrap();

        let err = context.read_tensor_f32("synthetic.weight").unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!(
                "Failed to read GGUF tensor synthetic.weight from {}: failed to fill whole buffer",
                fixture.path.display()
            )
        );
    }

    #[test]
    fn read_tensor_f32_reports_a_file_removed_after_opening() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(
                b"synthetic.weight",
                &[2],
                llama_cpp_bindings_sys::bindings::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0]),
            )
            .write("read_tensor_f32_reports_a_file_removed_after_opening");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        fs::remove_file(&fixture.path).unwrap();

        let result = context.read_tensor_f32("synthetic.weight");

        fs::write(&fixture.path, b"").unwrap();

        let err = result.unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!(
                "Failed to read GGUF tensor synthetic.weight from {}: No such file or directory (os error 2)",
                fixture.path.display()
            )
        );
    }

    #[test]
    fn reading_a_tensor_whose_name_has_a_nul_byte_is_refused() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.read_tensor_f32("foo\0bar").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "the GGUF tensor name to look up contains a NUL byte: nul byte found in provided data at position: 3"
        );
    }

    #[test]
    fn val_arr_i32_returns_the_stored_elements() {
        let fixture = SyntheticGgufBuilder::default()
            .value(
                b"synthetic.i32_array",
                GgufType::Array,
                &i32_array_bytes(&[3, -1, 7]),
            )
            .write("val_arr_i32_returns_the_stored_elements");
        let context = GgufContext::from_file(&fixture.path).unwrap();
        let array_index = context.find_key("synthetic.i32_array").unwrap();

        assert_eq!(context.val_arr_i32(array_index).unwrap(), &[3, -1, 7]);
    }

    #[test]
    fn val_arr_i32_rejects_a_key_that_holds_no_array() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let err = context.val_arr_i32(index).unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!("GGUF key id {index} holds a String value, not Array")
        );
    }

    #[test]
    fn val_arr_i32_rejects_an_array_of_another_element_type() {
        let mut encoded = GgufType::Uint8.to_raw().to_le_bytes().to_vec();

        encoded.extend_from_slice(&1u64.to_le_bytes());
        encoded.push(9);

        let fixture = SyntheticGgufBuilder::default()
            .value(b"synthetic.u8_array", GgufType::Array, &encoded)
            .write("val_arr_i32_rejects_an_array_of_another_element_type");
        let context = GgufContext::from_file(&fixture.path).unwrap();
        let array_index = context.find_key("synthetic.u8_array").unwrap();
        let err = context.val_arr_i32(array_index).unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!("GGUF key id {array_index} holds an array of type 0, not Int32")
        );
    }

    #[test]
    fn remove_key_rejects_a_missing_key() {
        let mut context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context.remove_key("nonexistent.key").unwrap_err();

        assert_eq!(
            error_chain(&err),
            "Key not found in GGUF context: nonexistent.key"
        );
    }

    #[test]
    fn write_to_file_keeps_every_tensor_and_key_except_the_removed_one() {
        let fixture = SyntheticGgufBuilder::default()
            .string_value(b"synthetic.kept", b"kept value")
            .string_value(b"synthetic.removed", b"removed value")
            .tensor(
                b"synthetic.weight",
                &[3, 2],
                llama_cpp_bindings_sys::bindings::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )
            .write("write_to_file_keeps_every_tensor_and_key_except_the_removed_one");
        let rewritten_path =
            rewritten_gguf_path("write_to_file_keeps_every_tensor_and_key_except_the_removed_one");
        let mut context = GgufContext::from_file(&fixture.path).unwrap();

        context.remove_key("synthetic.removed").unwrap();
        context.write_to_file(&rewritten_path).unwrap();

        let rewritten = GgufContext::from_file(&rewritten_path).unwrap();
        let kept_index = rewritten.find_key("synthetic.kept").unwrap();
        let removed_err = rewritten.find_key("synthetic.removed").unwrap_err();
        let tensor = rewritten.read_tensor_f32("synthetic.weight").unwrap();

        fs::remove_file(&rewritten_path).unwrap();

        assert_eq!(rewritten.n_kv(), 1);
        assert_eq!(rewritten.val_str(kept_index).unwrap(), "kept value");
        assert_eq!(
            error_chain(&removed_err),
            "Key not found in GGUF context: synthetic.removed"
        );
        assert_eq!(tensor.shape, [3, 2, 1, 1]);
        assert_eq!(tensor.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn write_to_file_holds_the_values_set_after_loading() {
        let fixture = SyntheticGgufBuilder::default()
            .value(
                b"synthetic.u32_value",
                GgufType::Uint32,
                &7u32.to_le_bytes(),
            )
            .write("write_to_file_holds_the_values_set_after_loading");
        let rewritten_path =
            rewritten_gguf_path("write_to_file_holds_the_values_set_after_loading");
        let mut context = GgufContext::from_file(&fixture.path).unwrap();

        context.set_val_u32(c"synthetic.u32_value", 4096);
        context.set_val_str(c"synthetic.str_value", c"set value");
        context.set_arr_i32(c"synthetic.i32_array", &[5, -8]);
        context.write_to_file(&rewritten_path).unwrap();

        let rewritten = GgufContext::from_file(&rewritten_path).unwrap();
        let u32_index = rewritten.find_key("synthetic.u32_value").unwrap();
        let str_index = rewritten.find_key("synthetic.str_value").unwrap();
        let array_index = rewritten.find_key("synthetic.i32_array").unwrap();

        fs::remove_file(&rewritten_path).unwrap();

        assert_eq!(rewritten.val_u32(u32_index).unwrap(), 4096);
        assert_eq!(rewritten.val_str(str_index).unwrap(), "set value");
        assert_eq!(rewritten.val_arr_i32(array_index).unwrap(), &[5, -8]);
    }

    #[test]
    fn write_to_file_reports_a_source_file_removed_after_opening() {
        let fixture = SyntheticGgufBuilder::default()
            .string_value(b"synthetic.kept", b"kept value")
            .write("write_to_file_reports_a_source_file_removed_after_opening");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        fs::remove_file(&fixture.path).unwrap();

        let result = context.write_to_file(rewritten_gguf_path(
            "write_to_file_reports_a_source_file_removed_after_opening",
        ));

        fs::write(&fixture.path, b"").unwrap();

        let err = result.unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!(
                "Failed to read the GGUF file {} this context was loaded from: No such file or directory (os error 2)",
                fixture.path.display()
            )
        );
    }

    #[test]
    fn write_to_file_reports_a_source_file_truncated_before_its_tensor_data() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(
                b"synthetic.weight",
                &[2],
                llama_cpp_bindings_sys::bindings::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0]),
            )
            .write("write_to_file_reports_a_source_file_truncated_before_its_tensor_data");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        fs::write(&fixture.path, b"").unwrap();

        let err = context
            .write_to_file(rewritten_gguf_path(
                "write_to_file_reports_a_source_file_truncated_before_its_tensor_data",
            ))
            .unwrap_err();

        assert_eq!(
            error_chain(&err),
            format!(
                "GGUF file {} holds 0 bytes, but its tensor data begins at byte {}",
                fixture.path.display(),
                unsafe {
                    llama_cpp_bindings_sys::bindings::gguf_get_data_offset(context.context.as_ptr())
                }
            )
        );
    }

    #[test]
    fn write_to_file_reports_a_destination_it_cannot_write() {
        let context = GgufContext::from_file(vocabulary_fixture_path()).unwrap();
        let err = context
            .write_to_file("/nonexistent/directory/rewritten.gguf")
            .unwrap_err();

        assert_eq!(
            error_chain(&err),
            "Failed to write the rewritten GGUF file to /nonexistent/directory/rewritten.gguf: No such file or directory (os error 2)"
        );
    }
}
