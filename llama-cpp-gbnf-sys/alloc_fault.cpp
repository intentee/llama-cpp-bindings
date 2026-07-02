#include "alloc_fault.h"

#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

thread_local bool alloc_fault_armed = false;
thread_local std::size_t alloc_fault_ops_to_skip = 0;
thread_local int alloc_fault_region_depth = 0;
thread_local bool alloc_fault_current_op_fails = false;

void * alloc_fault_allocate(std::size_t size) {
    if (alloc_fault_armed && alloc_fault_region_depth > 0 && alloc_fault_current_op_fails) {
        throw std::bad_alloc();
    }

    void * pointer = std::malloc(size == 0 ? 1 : size);

    if (pointer == nullptr) {
        throw std::bad_alloc();
    }

    return pointer;
}

} // namespace

extern "C" void gbnf_alloc_fault_arm(std::size_t skip_ops) {
    alloc_fault_armed = true;
    alloc_fault_ops_to_skip = skip_ops;
}

extern "C" void gbnf_alloc_fault_disarm(void) {
    alloc_fault_armed = false;
    alloc_fault_ops_to_skip = 0;
    alloc_fault_current_op_fails = false;
}

extern "C" void gbnf_alloc_fault_region_enter(void) {
    if (alloc_fault_region_depth == 0 && alloc_fault_armed) {
        if (alloc_fault_ops_to_skip > 0) {
            alloc_fault_ops_to_skip -= 1;
            alloc_fault_current_op_fails = false;
        } else {
            alloc_fault_current_op_fails = true;
        }
    }

    alloc_fault_region_depth += 1;
}

extern "C" void gbnf_alloc_fault_region_exit(void) {
    alloc_fault_region_depth -= 1;

    if (alloc_fault_region_depth == 0) {
        alloc_fault_current_op_fails = false;
    }
}

void * operator new(std::size_t size) {
    return alloc_fault_allocate(size);
}

void * operator new[](std::size_t size) {
    return alloc_fault_allocate(size);
}

void operator delete(void * pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void * pointer) noexcept {
    std::free(pointer);
}

void operator delete(void * pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void * pointer, std::size_t) noexcept {
    std::free(pointer);
}
