cmake_minimum_required(VERSION 3.14)
project(llama_cpp_bindings_build_fixture C)

if(NOT DEFINED FIXTURE_LIBRARY_NAME)
    message(FATAL_ERROR "FIXTURE_LIBRARY_NAME must be defined")
endif()

add_library(${FIXTURE_LIBRARY_NAME} STATIC fixture.c)

install(TARGETS ${FIXTURE_LIBRARY_NAME} DESTINATION lib)
