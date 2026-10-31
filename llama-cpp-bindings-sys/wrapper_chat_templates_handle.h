#pragma once

#include <common/chat.h>

struct llama_rs_chat_templates {
    common_chat_templates_ptr templates;
    bool supports_tool_calls = false;
};
