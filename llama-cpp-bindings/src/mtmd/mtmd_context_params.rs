use std::ffi::{CStr, CString};

use super::mtmd_context_params_error::MtmdContextParamsError;
use super::mtmd_image_token_limit::MtmdImageTokenLimit;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtmdContextParams {
    pub use_gpu: bool,
    pub print_timings: bool,
    pub n_threads: i32,
    pub media_marker: CString,
    pub image_min_tokens: MtmdImageTokenLimit,
    pub image_max_tokens: MtmdImageTokenLimit,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        unsafe { llama_cpp_bindings_sys::mtmd_context_params_default() }.into()
    }
}

impl TryFrom<&MtmdContextParams> for llama_cpp_bindings_sys::mtmd_context_params {
    type Error = MtmdContextParamsError;

    fn try_from(params: &MtmdContextParams) -> Result<Self, Self::Error> {
        let mut context = unsafe { llama_cpp_bindings_sys::mtmd_context_params_default() };
        let MtmdContextParams {
            use_gpu,
            print_timings,
            n_threads,
            media_marker,
            image_min_tokens,
            image_max_tokens,
        } = params;

        context.use_gpu = *use_gpu;
        context.print_timings = *print_timings;
        context.n_threads = *n_threads;
        context.media_marker = media_marker.as_ptr();
        context.image_min_tokens = image_min_tokens.to_c_int().map_err(|source| {
            MtmdContextParamsError::ImageMinTokensExceedCInt {
                limit: *image_min_tokens,
                source,
            }
        })?;
        context.image_max_tokens = image_max_tokens.to_c_int().map_err(|source| {
            MtmdContextParamsError::ImageMaxTokensExceedCInt {
                limit: *image_max_tokens,
                source,
            }
        })?;

        Ok(context)
    }
}

impl From<llama_cpp_bindings_sys::mtmd_context_params> for MtmdContextParams {
    fn from(params: llama_cpp_bindings_sys::mtmd_context_params) -> Self {
        Self {
            use_gpu: params.use_gpu,
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker: unsafe { CStr::from_ptr(params.media_marker) }.to_owned(),
            image_min_tokens: MtmdImageTokenLimit::from_c_int(params.image_min_tokens),
            image_max_tokens: MtmdImageTokenLimit::from_c_int(params.image_max_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::MtmdContextParams;
    use crate::mtmd::mtmd_context_params_error::MtmdContextParamsError;
    use crate::mtmd::mtmd_image_token_limit::MtmdImageTokenLimit;

    const SIXTEEN: NonZeroU32 = NonZeroU32::new(16).unwrap();
    const SIXTY_FOUR: NonZeroU32 = NonZeroU32::new(64).unwrap();
    const UNREPRESENTABLE_LIMIT: MtmdImageTokenLimit = MtmdImageTokenLimit::Tokens(NonZeroU32::MAX);

    #[test]
    fn image_token_limits_reach_the_c_parameters() {
        let params = MtmdContextParams {
            image_min_tokens: MtmdImageTokenLimit::Tokens(SIXTEEN),
            image_max_tokens: MtmdImageTokenLimit::Tokens(SIXTY_FOUR),
            ..MtmdContextParams::default()
        };

        assert_eq!(
            llama_cpp_bindings_sys::mtmd_context_params::try_from(&params)
                .map(|c_params| [c_params.image_min_tokens, c_params.image_max_tokens]),
            Ok([16, 64])
        );
    }

    #[test]
    fn an_unrepresentable_image_min_token_limit_is_refused() {
        let params = MtmdContextParams {
            image_min_tokens: UNREPRESENTABLE_LIMIT,
            ..MtmdContextParams::default()
        };

        assert_eq!(
            llama_cpp_bindings_sys::mtmd_context_params::try_from(&params).err(),
            i32::try_from(u32::MAX).err().map(|source| {
                MtmdContextParamsError::ImageMinTokensExceedCInt {
                    limit: UNREPRESENTABLE_LIMIT,
                    source,
                }
            })
        );
    }

    #[test]
    fn an_unrepresentable_image_max_token_limit_is_refused() {
        let params = MtmdContextParams {
            image_max_tokens: UNREPRESENTABLE_LIMIT,
            ..MtmdContextParams::default()
        };

        assert_eq!(
            llama_cpp_bindings_sys::mtmd_context_params::try_from(&params).err(),
            i32::try_from(u32::MAX).err().map(|source| {
                MtmdContextParamsError::ImageMaxTokensExceedCInt {
                    limit: UNREPRESENTABLE_LIMIT,
                    source,
                }
            })
        );
    }
}
