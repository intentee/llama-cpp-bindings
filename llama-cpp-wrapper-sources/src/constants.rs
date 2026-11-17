pub const CPP_STANDARD: &str = "c++17";

pub const FAULT_INJECTION_DEFINE: &str = "LLAMA_RS_FAULT_INJECTION";

pub const VENDORED_INCLUDE_DIRS: &[&str] = &[
    "GSL/include",
    "llama.cpp",
    "llama.cpp/common",
    "llama.cpp/include",
    "llama.cpp/ggml/include",
    "llama.cpp/vendor",
];

pub const WRAPPER_HEADERS: &[&str] = &[
    "wrapper.h",
    "wrapper_batch.h",
    "wrapper_chat_session.h",
    "wrapper_chat_templates.h",
    "wrapper_chat_templates_handle.h",
    "wrapper_common.h",
    "wrapper_context.h",
    "wrapper_decision.h",
    "wrapper_fault_injection.h",
    "wrapper_fit.h",
    "wrapper_gbnf.h",
    "wrapper_mtmd.h",
    "wrapper_parsed_chat.h",
    "wrapper_parsed_chat_handle.h",
    "wrapper_report_current_exception.h",
    "wrapper_sampler_init.h",
    "wrapper_state.h",
    "wrapper_utils.h",
];

pub const WRAPPER_INCLUDE_DIRS: &[&str] = &["."];

pub const WRAPPER_SOURCES: &[&str] = &[
    "wrapper_batch.cpp",
    "wrapper_chat_session.cpp",
    "wrapper_chat_templates.cpp",
    "wrapper_common.cpp",
    "wrapper_context.cpp",
    "wrapper_decision.cpp",
    "wrapper_fault_injection.cpp",
    "wrapper_fit.cpp",
    "wrapper_gbnf.cpp",
    "wrapper_mtmd.cpp",
    "wrapper_parsed_chat.cpp",
    "wrapper_sampler_init.cpp",
    "wrapper_state.cpp",
];
