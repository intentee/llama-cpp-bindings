#include <vendored.h>

#ifndef WRAPPER_FIXTURE_DEFINE
#error "WRAPPER_FIXTURE_DEFINE must be defined"
#endif

extern "C" auto llama_cpp_bindings_build_wrapper_fixture() -> int {
    return VENDORED_VALUE;
}
