#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod model_provided_tool_call_id;
pub mod model_provided_tool_call_id_error;
pub mod parsed_chat_message;
pub mod parsed_tool_call;
pub mod parsed_tool_call_id;
pub mod token_usage;
pub mod token_usage_error;
pub mod tool_call_arguments;
