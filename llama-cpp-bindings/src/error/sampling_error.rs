#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error("a value does not fit into i32")]
    IntegerOverflow(#[source] std::num::TryFromIntError),
    #[error("{sampler} sampler could not be initialized")]
    SamplerUnavailable { sampler: &'static str },
}
