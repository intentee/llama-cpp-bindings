use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::slice;

use crate::gguf_context_error::GgufContextError;
use crate::gguf_tensor_f32::GgufTensorF32;
use crate::gguf_type::GgufType;

const F32_SIZE_IN_BYTES: usize = size_of::<f32>();

#[derive(Debug)]
pub struct GgufContext {
    context: NonNull<llama_cpp_bindings_sys::gguf_context>,
    path: PathBuf,
}

impl GgufContext {
    /// # Errors
    ///
    /// Returns [`GgufContextError::InitFailed`] if the file cannot be opened or parsed.
    /// Returns [`GgufContextError::PathToStrError`] if the path is not valid UTF-8.
    /// Returns [`GgufContextError::NulError`] if the path contains a null byte.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, GgufContextError> {
        let path_ref = path.as_ref();
        let path_str = path_ref
            .to_str()
            .ok_or_else(|| GgufContextError::PathToStrError(path_ref.to_path_buf()))?;
        let c_path = CString::new(path_str)?;

        let init_params = llama_cpp_bindings_sys::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };

        let raw =
            unsafe { llama_cpp_bindings_sys::gguf_init_from_file(c_path.as_ptr(), init_params) };
        let context = NonNull::new(raw)
            .ok_or_else(|| GgufContextError::InitFailed(path_ref.to_path_buf()))?;

        Ok(Self {
            context,
            path: path_ref.to_path_buf(),
        })
    }

    #[must_use]
    pub fn n_kv(&self) -> i64 {
        unsafe { llama_cpp_bindings_sys::gguf_get_n_kv(self.context.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyNotFound`] if the key does not exist.
    /// Returns [`GgufContextError::NulError`] if the key contains a null byte.
    pub fn find_key(&self, key: &str) -> Result<i64, GgufContextError> {
        let c_key = CString::new(key)?;
        let index =
            unsafe { llama_cpp_bindings_sys::gguf_find_key(self.context.as_ptr(), c_key.as_ptr()) };

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
    /// Returns [`GgufContextError::Utf8Error`] if the key name is not valid UTF-8.
    pub fn key_at(&self, key_id: i64) -> Result<&str, GgufContextError> {
        self.require_key_id_in_range(key_id)?;

        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::gguf_get_key(
                self.context.as_ptr(),
                key_id,
            ))
        };

        Ok(c_str.to_str()?)
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] if the file holds no key with this id.
    /// Returns [`GgufContextError::UnknownValueType`] if the value type is not a known GGUF type.
    pub fn kv_type(&self, key_id: i64) -> Result<GgufType, GgufContextError> {
        self.require_key_id_in_range(key_id)?;

        let raw_type =
            unsafe { llama_cpp_bindings_sys::gguf_get_kv_type(self.context.as_ptr(), key_id) };

        GgufType::from_raw(raw_type).ok_or(GgufContextError::UnknownValueType { key_id, raw_type })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Uint32`] value.
    pub fn val_u32(&self, key_id: i64) -> Result<u32, GgufContextError> {
        self.require_value_type(key_id, GgufType::Uint32)?;

        Ok(unsafe { llama_cpp_bindings_sys::gguf_get_val_u32(self.context.as_ptr(), key_id) })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Int32`] value.
    pub fn val_i32(&self, key_id: i64) -> Result<i32, GgufContextError> {
        self.require_value_type(key_id, GgufType::Int32)?;

        Ok(unsafe { llama_cpp_bindings_sys::gguf_get_val_i32(self.context.as_ptr(), key_id) })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Uint64`] value.
    pub fn val_u64(&self, key_id: i64) -> Result<u64, GgufContextError> {
        self.require_value_type(key_id, GgufType::Uint64)?;

        Ok(unsafe { llama_cpp_bindings_sys::gguf_get_val_u64(self.context.as_ptr(), key_id) })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::Float32`] value.
    pub fn val_f32(&self, key_id: i64) -> Result<f32, GgufContextError> {
        self.require_value_type(key_id, GgufType::Float32)?;

        Ok(unsafe { llama_cpp_bindings_sys::gguf_get_val_f32(self.context.as_ptr(), key_id) })
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyIdOutOfRange`] or [`GgufContextError::ValueTypeMismatch`]
    /// if the key does not hold a [`GgufType::String`] value.
    /// Returns [`GgufContextError::Utf8Error`] if the string value is not valid UTF-8.
    pub fn val_str(&self, key_id: i64) -> Result<&str, GgufContextError> {
        self.require_value_type(key_id, GgufType::String)?;

        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::gguf_get_val_str(
                self.context.as_ptr(),
                key_id,
            ))
        };

        Ok(c_str.to_str()?)
    }

    #[must_use]
    pub fn n_tensors(&self) -> i64 {
        unsafe { llama_cpp_bindings_sys::gguf_get_n_tensors(self.context.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns [`GgufContextError::TensorNotFound`] if the file holds no tensor with this name.
    /// Returns [`GgufContextError::TensorIsNotF32`] if the tensor is stored in another type.
    /// Returns [`GgufContextError::TensorReadFailed`] if the tensor data cannot be read.
    /// Returns [`GgufContextError::NulError`] if the name contains a null byte.
    pub fn read_tensor_f32(&self, name: &str) -> Result<GgufTensorF32, GgufContextError> {
        let c_name = CString::new(name)?;
        let tensor_id = unsafe {
            llama_cpp_bindings_sys::gguf_find_tensor(self.context.as_ptr(), c_name.as_ptr())
        };

        if tensor_id < 0 {
            return Err(GgufContextError::TensorNotFound {
                name: name.to_owned(),
            });
        }

        let ggml_type = unsafe {
            llama_cpp_bindings_sys::gguf_get_tensor_type(self.context.as_ptr(), tensor_id)
        };

        if ggml_type != llama_cpp_bindings_sys::GGML_TYPE_F32 {
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

    fn tensor_shape(
        &self,
        tensor_id: i64,
    ) -> [i64; llama_cpp_bindings_sys::GGML_MAX_DIMS as usize] {
        let element_counts = unsafe {
            slice::from_raw_parts(
                llama_cpp_bindings_sys::gguf_get_tensor_ne(self.context.as_ptr(), tensor_id),
                llama_cpp_bindings_sys::GGML_MAX_DIMS as usize,
            )
        };
        let mut shape = [0; llama_cpp_bindings_sys::GGML_MAX_DIMS as usize];

        shape.copy_from_slice(element_counts);

        shape
    }

    fn tensor_f32_values(&self, name: &str, tensor_id: i64) -> Result<Vec<f32>, GgufContextError> {
        let data_offset =
            unsafe { llama_cpp_bindings_sys::gguf_get_data_offset(self.context.as_ptr()) };
        let tensor_offset = unsafe {
            llama_cpp_bindings_sys::gguf_get_tensor_offset(self.context.as_ptr(), tensor_id)
        };
        let tensor_size = unsafe {
            llama_cpp_bindings_sys::gguf_get_tensor_size(self.context.as_ptr(), tensor_id)
        };
        let read_failed = |source: io::Error| GgufContextError::TensorReadFailed {
            name: name.to_owned(),
            path: self.path.clone(),
            source,
        };

        let file = File::open(&self.path).map_err(read_failed)?;
        let mut bytes = vec![0; tensor_size];

        file.read_exact_at(&mut bytes, (data_offset + tensor_offset) as u64)
            .map_err(read_failed)?;

        Ok(bytes
            .chunks_exact(F32_SIZE_IN_BYTES)
            .map(|value_bytes| {
                f32::from_ne_bytes([
                    value_bytes[0],
                    value_bytes[1],
                    value_bytes[2],
                    value_bytes[3],
                ])
            })
            .collect())
    }
}

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::gguf_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::mem::Discriminant;
    use std::path::PathBuf;

    use super::GgufContext;
    use crate::gguf_context_error::GgufContextError;
    use crate::gguf_type::GgufType;

    const GGUF_ALIGNMENT: usize = 32;
    const GGML_TYPE_F16: u32 = 1;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("ggml-vocab-bert-bge.gguf")
    }

    fn init_failed_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::InitFailed(PathBuf::new()))
    }

    fn key_not_found_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::KeyNotFound { key: String::new() })
    }

    fn nul_error_disc() -> Discriminant<GgufContextError> {
        let nul_err = CString::new(b"a\0b".to_vec()).unwrap_err();
        std::mem::discriminant(&GgufContextError::NulError(nul_err))
    }

    fn path_to_str_error_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::PathToStrError(PathBuf::new()))
    }

    fn key_id_out_of_range_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::KeyIdOutOfRange { key_id: 0, n_kv: 0 })
    }

    fn value_type_mismatch_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::ValueTypeMismatch {
            key_id: 0,
            expected: GgufType::Uint32,
            actual: GgufType::Uint32,
        })
    }

    fn tensor_not_found_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::TensorNotFound {
            name: String::new(),
        })
    }

    fn tensor_is_not_f32_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::TensorIsNotF32 {
            name: String::new(),
            ggml_type: 0,
        })
    }

    fn tensor_read_failed_disc() -> Discriminant<GgufContextError> {
        std::mem::discriminant(&GgufContextError::TensorReadFailed {
            name: String::new(),
            path: PathBuf::new(),
            source: std::io::Error::other("discriminant"),
        })
    }

    fn utf8_error_disc() -> Discriminant<GgufContextError> {
        let invalid_utf8_bytes: Vec<u8> = vec![0xFF];
        let utf8_err = std::str::from_utf8(&invalid_utf8_bytes).unwrap_err();
        std::mem::discriminant(&GgufContextError::Utf8Error(utf8_err))
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
        fn value(mut self, key: &[u8], gguf_type: GgufType, value: &[u8]) -> Self {
            self.key_values
                .extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.key_values.extend_from_slice(key);
            self.key_values
                .extend_from_slice(&gguf_type.to_raw().to_le_bytes());
            self.key_values.extend_from_slice(value);
            self.key_value_count += 1;
            self
        }

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
            use std::io::Write as _;

            let path = std::env::temp_dir().join(format!(
                "llama_cpp_bindings_synthetic_{}_{}.gguf",
                std::process::id(),
                test_name,
            ));

            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(bytes).unwrap();

            Self { path }
        }
    }

    impl Drop for SyntheticGgufFile {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path)
                .unwrap_or_else(|error| panic!("failed to remove synthetic GGUF: {error}"));
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn from_file_opens_valid_gguf() {
        let context = GgufContext::from_file(fixture_path());

        assert!(context.is_ok());
    }

    #[test]
    fn from_file_nonexistent_returns_init_failed() {
        let err = GgufContext::from_file("/nonexistent/file.gguf").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), init_failed_disc());
    }

    #[test]
    fn n_kv_returns_positive_count() {
        let context = GgufContext::from_file(fixture_path()).unwrap();

        assert!(context.n_kv() > 0);
    }

    #[test]
    fn find_key_returns_valid_index_for_known_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture");

        assert!(index.is_ok());
        assert!(index.unwrap() >= 0);
    }

    #[test]
    fn find_key_returns_error_for_missing_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let err = context.find_key("nonexistent.key").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), key_not_found_disc());
    }

    #[test]
    fn key_at_returns_expected_name() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let key_name = context.key_at(index).unwrap();

        assert_eq!(key_name, "general.architecture");
    }

    #[test]
    fn key_at_rejects_a_key_id_past_the_last_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let err = context.key_at(context.n_kv()).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), key_id_out_of_range_disc());
    }

    #[test]
    fn typed_value_rejects_a_negative_key_id() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let err = context.val_u32(-1).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), key_id_out_of_range_disc());
    }

    #[test]
    fn kv_type_returns_expected_type_for_string_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();

        assert_eq!(context.kv_type(index).unwrap(), GgufType::String);
    }

    #[test]
    fn val_str_returns_architecture_value() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let value = context.val_str(index).unwrap();

        assert!(!value.is_empty());
    }

    #[test]
    fn from_file_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
        let err = GgufContext::from_file(non_utf8_path).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), path_to_str_error_disc());
    }

    #[test]
    fn from_file_with_null_byte_in_path_returns_error() {
        let err = GgufContext::from_file("/tmp/foo\0bar.gguf").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), nul_error_disc());
    }

    #[test]
    fn find_key_with_null_byte_in_key_returns_error() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let err = context.find_key("foo\0bar").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), nul_error_disc());
    }

    #[test]
    fn typed_values_round_trip_through_synthetic_fixture() {
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

        for mismatch in mismatches {
            assert_eq!(
                std::mem::discriminant(&mismatch),
                value_type_mismatch_disc()
            );
        }
    }

    #[test]
    fn val_str_returns_utf8_error_for_non_utf8_value() {
        let fixture = SyntheticGgufBuilder::default()
            .string_value(b"synthetic.str_value", &[0xFF, 0xFE])
            .write("val_str_returns_utf8_error_for_non_utf8_value");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let value_index = context.find_key("synthetic.str_value").unwrap();
        let err = context.val_str(value_index).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), utf8_error_disc());
    }

    #[test]
    fn key_at_returns_utf8_error_for_non_utf8_key() {
        let fixture = SyntheticGgufBuilder::default()
            .value(&[0xFF, 0xFE], GgufType::Int32, &42i32.to_le_bytes())
            .write("key_at_returns_utf8_error_for_non_utf8_key");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let err = context.key_at(0).unwrap_err();

        assert_eq!(std::mem::discriminant(&err), utf8_error_disc());
    }

    #[test]
    fn read_tensor_f32_returns_shape_and_values() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(b"synthetic.bias", &[2], GGML_TYPE_F16, vec![0; 4])
            .tensor(
                b"synthetic.weight",
                &[3, 2],
                llama_cpp_bindings_sys::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )
            .write("read_tensor_f32_returns_shape_and_values");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        let tensor = context.read_tensor_f32("synthetic.weight").unwrap();

        assert_eq!(context.n_tensors(), 2);
        assert_eq!(tensor.shape, [3, 2, 1, 1]);
        assert_eq!(tensor.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn read_tensor_f32_rejects_a_missing_tensor() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let err = context.read_tensor_f32("synthetic.missing").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), tensor_not_found_disc());
    }

    #[test]
    fn read_tensor_f32_rejects_a_tensor_stored_in_another_type() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(b"synthetic.half", &[2], GGML_TYPE_F16, vec![0; 4])
            .write("read_tensor_f32_rejects_another_type");
        let context = GgufContext::from_file(&fixture.path).unwrap();
        let err = context.read_tensor_f32("synthetic.half").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), tensor_is_not_f32_disc());
    }

    #[test]
    fn read_tensor_f32_reports_a_file_truncated_after_opening() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(
                b"synthetic.weight",
                &[2],
                llama_cpp_bindings_sys::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0]),
            )
            .write("read_tensor_f32_reports_a_file_truncated_after_opening");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        std::fs::write(&fixture.path, b"").unwrap();

        let err = context.read_tensor_f32("synthetic.weight").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), tensor_read_failed_disc());
    }

    #[test]
    fn read_tensor_f32_reports_a_file_removed_after_opening() {
        let fixture = SyntheticGgufBuilder::default()
            .tensor(
                b"synthetic.weight",
                &[2],
                llama_cpp_bindings_sys::GGML_TYPE_F32,
                f32_bytes(&[1.0, 2.0]),
            )
            .write("read_tensor_f32_reports_a_file_removed_after_opening");
        let context = GgufContext::from_file(&fixture.path).unwrap();

        std::fs::remove_file(&fixture.path).unwrap();

        let result = context.read_tensor_f32("synthetic.weight");

        std::fs::write(&fixture.path, b"").unwrap();

        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            tensor_read_failed_disc()
        );
    }

    #[test]
    fn read_tensor_f32_with_null_byte_in_name_returns_error() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let err = context.read_tensor_f32("foo\0bar").unwrap_err();

        assert_eq!(std::mem::discriminant(&err), nul_error_disc());
    }
}
