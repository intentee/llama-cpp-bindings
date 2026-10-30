use crate::LogitsError;
use crate::context::decoded_output_rows::DecodedOutputRows;
use crate::context::llama_pooling_type::LlamaPoolingType;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedOutputs {
    pub rows: DecodedOutputRows,
    pub pooling_type: LlamaPoolingType,
}

impl DecodedOutputs {
    /// Accepts the batch index of an output token, or a negative index counting back from the
    /// last output, the way `llama_get_logits_ith` resolves it.
    ///
    /// # Errors
    ///
    /// Returns [`LogitsError::TokenNotInitialized`] when the index names no output of the last
    /// decode.
    pub fn ensure_output(&self, token_index: i32) -> Result<(), LogitsError> {
        let names_an_output = match &self.rows {
            DecodedOutputRows::BatchIndices(output_token_indices) if token_index >= 0 => {
                output_token_indices.contains(&token_index)
            }
            DecodedOutputRows::BatchIndices(output_token_indices) => {
                token_index.unsigned_abs() as usize <= output_token_indices.len()
            }
            DecodedOutputRows::LastRow => token_index == -1,
        };

        if names_an_output {
            Ok(())
        } else {
            Err(LogitsError::TokenNotInitialized(token_index))
        }
    }

    #[must_use]
    pub const fn missing_logits_error(&self) -> LogitsError {
        match self.pooling_type {
            LlamaPoolingType::None | LlamaPoolingType::Unspecified => {
                LogitsError::GraphProducesNoLogits
            }
            pooling_type @ (LlamaPoolingType::Mean
            | LlamaPoolingType::Cls
            | LlamaPoolingType::Last
            | LlamaPoolingType::Rank) => {
                LogitsError::PooledEmbeddingsProduceNoLogits { pooling_type }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DecodedOutputs;
    use crate::LogitsError;
    use crate::context::decoded_output_rows::DecodedOutputRows;
    use crate::context::llama_pooling_type::LlamaPoolingType;

    fn outputs_at(output_token_indices: Vec<i32>) -> DecodedOutputs {
        DecodedOutputs {
            rows: DecodedOutputRows::BatchIndices(output_token_indices),
            pooling_type: LlamaPoolingType::None,
        }
    }

    fn last_row_only() -> DecodedOutputs {
        DecodedOutputs {
            rows: DecodedOutputRows::LastRow,
            pooling_type: LlamaPoolingType::None,
        }
    }

    #[test]
    fn a_batch_index_marked_as_output_names_an_output() {
        assert_eq!(outputs_at(vec![2, 5]).ensure_output(5), Ok(()));
    }

    #[test]
    fn a_batch_index_not_marked_as_output_is_refused() {
        assert_eq!(
            outputs_at(vec![2, 5]).ensure_output(3),
            Err(LogitsError::TokenNotInitialized(3))
        );
    }

    #[test]
    fn a_negative_index_counts_back_from_the_last_output() {
        assert_eq!(outputs_at(vec![2, 5]).ensure_output(-2), Ok(()));
    }

    #[test]
    fn a_negative_index_beyond_the_first_output_is_refused() {
        assert_eq!(
            outputs_at(vec![2, 5]).ensure_output(-3),
            Err(LogitsError::TokenNotInitialized(-3))
        );
    }

    #[test]
    fn a_decode_outputting_only_its_last_row_names_it_from_the_end() {
        assert_eq!(last_row_only().ensure_output(-1), Ok(()));
    }

    #[test]
    fn a_decode_outputting_only_its_last_row_refuses_batch_indices() {
        assert_eq!(
            last_row_only().ensure_output(0),
            Err(LogitsError::TokenNotInitialized(0))
        );
    }

    #[test]
    fn missing_logits_of_a_pooled_decode_are_attributed_to_pooling() {
        let decoded = DecodedOutputs {
            rows: DecodedOutputRows::BatchIndices(vec![0]),
            pooling_type: LlamaPoolingType::Mean,
        };

        assert_eq!(
            decoded.missing_logits_error(),
            LogitsError::PooledEmbeddingsProduceNoLogits {
                pooling_type: LlamaPoolingType::Mean
            }
        );
    }

    #[test]
    fn missing_logits_of_an_unpooled_decode_are_attributed_to_the_graph() {
        assert_eq!(
            outputs_at(vec![0]).missing_logits_error(),
            LogitsError::GraphProducesNoLogits
        );
    }
}
