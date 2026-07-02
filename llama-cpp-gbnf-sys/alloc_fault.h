#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GBNF_FAULT_INJECTION

void gbnf_alloc_fault_arm(size_t skip_ops);
void gbnf_alloc_fault_disarm(void);
void gbnf_alloc_fault_region_enter(void);
void gbnf_alloc_fault_region_exit(void);

#else

static inline void gbnf_alloc_fault_region_enter(void) {}
static inline void gbnf_alloc_fault_region_exit(void) {}

#endif

#ifdef __cplusplus
}
#endif
