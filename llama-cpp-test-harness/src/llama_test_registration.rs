use inventory::collect;

use llama_cpp_test_model_provisioning::load_key::LoadKey;

use crate::context_params::ContextParams;
use crate::expected_outcome::ExpectedOutcome;
use crate::llama_test_fn::LlamaTestFn;

pub struct LlamaTestRegistration {
    pub name: &'static str,
    pub key: LoadKey,
    pub context_params: ContextParams,
    pub func: LlamaTestFn,
    pub expected_outcome: ExpectedOutcome,
}

collect!(LlamaTestRegistration);
