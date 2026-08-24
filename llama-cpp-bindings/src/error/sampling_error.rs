#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error("Integer overflow: {0}")]
    IntegerOverflow(String),
    #[error("{sampler} sampler could not be initialized")]
    SamplerUnavailable { sampler: &'static str },
}
