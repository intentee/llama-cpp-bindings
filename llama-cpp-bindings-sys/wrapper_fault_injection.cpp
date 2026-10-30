#include "wrapper_fault_injection.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef LLAMA_RS_FAULT_INJECTION
namespace {

constexpr const char * INJECTED_FAULT_MESSAGE = "llama-cpp-bindings injected a fault";

struct armed_fault {
    llama_rs_fault_point point = LLAMA_RS_FAULT_POINT_NONE;
    size_t passing_hits = 0;
};

auto armed_fault_on_this_thread() -> armed_fault & {
    thread_local armed_fault fault;

    return fault;
}

}  // namespace

extern "C" void llama_rs_fault_injection_arm(llama_rs_fault_point point, size_t passing_hits) {
    armed_fault_on_this_thread() = armed_fault{point, passing_hits};
}

extern "C" void llama_rs_fault_injection_disarm() {
    armed_fault_on_this_thread() = armed_fault{};
}

extern "C" auto llama_rs_fault_injection_message() -> const char * {
    return INJECTED_FAULT_MESSAGE;
}

auto llama_rs_fault_injected(llama_rs_fault_point point) -> bool {
    armed_fault & fault = armed_fault_on_this_thread();

    if (fault.point != point) {
        return false;
    }

    if (fault.passing_hits > 0) {
        fault.passing_hits -= 1;

        return false;
    }

    return true;
}

void llama_rs_throw_if_fault_injected(llama_rs_fault_point point) {
    if (llama_rs_fault_injected(point)) {
        throw std::runtime_error(INJECTED_FAULT_MESSAGE);
    }
}
#else
auto llama_rs_fault_injected([[maybe_unused]] llama_rs_fault_point point) -> bool {
    return false;
}

void llama_rs_throw_if_fault_injected([[maybe_unused]] llama_rs_fault_point point) {}
#endif

auto llama_rs_text_view_unless_fault_injected(
    llama_rs_fault_point point,
    const std::string & text,
    size_t * out_length) -> const char * {
    if (llama_rs_fault_injected(point)) {
        static const std::string not_utf8 = "\xff";

        return llama_rs_string_view(not_utf8, out_length);
    }

    return llama_rs_string_view(text, out_length);
}

extern "C" auto llama_rs_fault_point_reached(llama_rs_fault_point point) -> bool {
    return llama_rs_fault_injected(point);
}
