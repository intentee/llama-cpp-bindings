use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;

use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::execution_phase::ExecutionPhase;
use crate::harness_run_error::HarnessRunError;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::scheduled_phase::ScheduledPhase;

fn refuse_duplicate_trial_names(
    registrations: &[&'static LlamaTestRegistration],
) -> Result<(), HarnessRunError> {
    let mut trial_names = BTreeSet::new();

    registrations.iter().try_for_each(|registration| {
        if trial_names.insert(registration.name) {
            Ok(())
        } else {
            Err(HarnessRunError::DuplicateTrialName {
                name: registration.name,
            })
        }
    })
}

fn collect_inventory_registrations() -> Vec<&'static LlamaTestRegistration> {
    inventory::iter::<LlamaTestRegistration>
        .into_iter()
        .collect()
}

pub struct ExecutionPlan {
    pub phases: Vec<ExecutionPhase>,
}

impl ExecutionPlan {
    /// # Errors
    ///
    /// Returns [`HarnessRunError::DuplicateTrialName`] when two registrations share a trial name,
    /// since their results could not be told apart.
    pub fn from_registrations(
        registrations: &[&'static LlamaTestRegistration],
    ) -> Result<Self, HarnessRunError> {
        refuse_duplicate_trial_names(registrations)?;

        let mut by_key: BTreeMap<_, Vec<&'static LlamaTestRegistration>> = BTreeMap::new();
        for registration in registrations {
            by_key
                .entry(registration.key)
                .or_default()
                .push(*registration);
        }
        let mut phases = Vec::with_capacity(by_key.len());
        for (key, mut registrations) in by_key {
            registrations.sort_by_key(|registration| registration.name);
            phases.push(ExecutionPhase { key, registrations });
        }
        Ok(Self { phases })
    }

    /// # Errors
    ///
    /// Returns the [`HarnessRunError`] of [`Self::from_registrations`] for every registered trial.
    pub fn from_inventory() -> Result<Self, HarnessRunError> {
        Self::from_registrations(&collect_inventory_registrations())
    }

    #[must_use]
    pub fn run(&self, backend: &Arc<LlamaBackend>, arguments: &Arguments) -> Vec<Conclusion> {
        let total = self.phases.len();
        let scheduled_phases: Vec<ScheduledPhase<'_>> = self
            .phases
            .iter()
            .map(|phase| ScheduledPhase::new(phase, backend))
            .collect();

        for scheduled_phase in &scheduled_phases {
            scheduled_phase.download_model_files(arguments);
        }

        scheduled_phases
            .into_iter()
            .enumerate()
            .map(|(index, scheduled_phase)| scheduled_phase.run(index, total, arguments))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;
    use llama_cpp_test_model_provisioning::gguf_source::GgufSource;
    use llama_cpp_test_model_provisioning::load_key::LoadKey;
    use llama_cpp_test_model_provisioning::model_load_params::ModelLoadParams;

    use super::ExecutionPlan;
    use crate::context_params::ContextParams;
    use crate::expected_outcome::ExpectedOutcome;
    use crate::llama_test_fn::LlamaTestFn;
    use crate::llama_test_registration::LlamaTestRegistration;
    use crate::no_op::no_op;

    const TRIVIAL_CONTEXT_PARAMS: ContextParams = ContextParams {
        n_ctx: 1,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    const ALTERNATE_CONTEXT_PARAMS: ContextParams = ContextParams {
        n_ctx: 4096,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    static REG_BETA_A: LlamaTestRegistration = LlamaTestRegistration {
        name: "alpha",
        key: LoadKey {
            model_source: GgufSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        func: LlamaTestFn::Infallible(no_op),
        expected_outcome: ExpectedOutcome::Success,
    };
    static REG_BETA_B: LlamaTestRegistration = LlamaTestRegistration {
        name: "bravo",
        key: LoadKey {
            model_source: GgufSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        func: LlamaTestFn::Infallible(no_op),
        expected_outcome: ExpectedOutcome::Success,
    };
    static REG_ALPHA_Z: LlamaTestRegistration = LlamaTestRegistration {
        name: "zulu",
        key: LoadKey {
            model_source: GgufSource::HuggingFace {
                repo: "alpha",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        },
        context_params: TRIVIAL_CONTEXT_PARAMS,
        func: LlamaTestFn::Infallible(no_op),
        expected_outcome: ExpectedOutcome::Success,
    };
    static REG_BETA_DIFFERENT_CONTEXT: LlamaTestRegistration = LlamaTestRegistration {
        name: "charlie",
        key: LoadKey {
            model_source: GgufSource::HuggingFace {
                repo: "beta",
                file: "f",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        },
        context_params: ALTERNATE_CONTEXT_PARAMS,
        func: LlamaTestFn::Infallible(no_op),
        expected_outcome: ExpectedOutcome::Success,
    };

    #[test]
    fn registrations_sharing_a_trial_name_are_refused() {
        assert_eq!(
            ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_BETA_A])
                .err()
                .map(|refusal| refusal.to_string()),
            Some(
                "two trials are named `alpha`; every trial needs its own name so its result \
                 can be told apart"
                    .to_owned()
            )
        );
    }

    #[test]
    fn from_registrations_with_empty_input_yields_empty_plan() {
        let plan = ExecutionPlan::from_registrations(&[])
            .expect("registrations with distinct names form a plan");

        assert!(plan.phases.is_empty());
    }

    #[test]
    fn registrations_with_same_load_key_collapse_to_one_phase() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_BETA_B])
            .expect("registrations with distinct names form a plan");

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].registrations.len(), 2);
    }

    #[test]
    fn registrations_with_distinct_load_keys_form_phases_in_load_key_sort_order() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_ALPHA_Z])
            .expect("registrations with distinct names form a plan");

        assert_eq!(plan.phases.len(), 2);
        assert_eq!(
            plan.phases[0].key.model_source,
            GgufSource::HuggingFace {
                repo: "alpha",
                file: "f"
            }
        );
        assert_eq!(
            plan.phases[1].key.model_source,
            GgufSource::HuggingFace {
                repo: "beta",
                file: "f"
            }
        );
    }

    #[test]
    fn within_a_phase_registrations_sort_by_name() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_B, &REG_BETA_A])
            .expect("registrations with distinct names form a plan");

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].registrations[0].name, "alpha");
        assert_eq!(plan.phases[0].registrations[1].name, "bravo");
    }

    #[test]
    fn registrations_sharing_a_load_key_but_differing_context_params_stay_in_one_phase() {
        let plan = ExecutionPlan::from_registrations(&[&REG_BETA_A, &REG_BETA_DIFFERENT_CONTEXT])
            .expect("registrations with distinct names form a plan");

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].registrations.len(), 2);
        let context_lengths: Vec<u32> = plan.phases[0]
            .registrations
            .iter()
            .map(|registration| registration.context_params.n_ctx)
            .collect();
        assert!(context_lengths.contains(&1));
        assert!(context_lengths.contains(&4096));
    }
}
