use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

use anyhow::Result;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::model_with_patched_metadata::model_with_patched_metadata;

/// Loads a copy of the fixture's model whose metadata declares `decision_type` as its decision
/// type.
///
/// # Errors
///
/// Fails when the model's architecture cannot be read, or the copy cannot be written or loaded.
pub fn model_declaring_decision_type(
    fixture: &LlamaFixture<'_>,
    decision_type: &CStr,
    file_name: &str,
) -> Result<Arc<LlamaModel>> {
    let decision_type_key = CString::new(format!(
        "{}.decision.type",
        fixture.model.meta_val_str("general.architecture")?
    ))?;

    model_with_patched_metadata(fixture, file_name, |gguf| {
        gguf.set_val_str(&decision_type_key, decision_type);

        Ok(())
    })
}
