#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_context;

bool llama_rs_context_decodes_batches_in_one_micro_batch(const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
