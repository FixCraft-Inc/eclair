# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.

# Presentation and argument handling, testable without any backend.
add_library(eclair_smi_core STATIC
    json_report.cpp
    options.cpp
    system_report.cpp
    terminal.cpp
    text_report.cpp
)
target_include_directories(eclair_smi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(eclair_smi_core PUBLIC eclair_capability)
eclair_target_warnings(eclair_smi_core)

# The build decides which backends the tool can probe.
if(TARGET eclair_cuda)
    set(eclair_smi_backends backends_cuda.cpp)
    set(eclair_smi_backend_libraries eclair_cuda)
else()
    set(eclair_smi_backends backends_host_only.cpp)
    set(eclair_smi_backend_libraries)
endif()

set(ECLAIR_SMI_DEFAULT_STYLE "plain" CACHE STRING
    "Default eclair-smi output style: plain (words only, like nvidia-smi) or pretty")
set_property(CACHE ECLAIR_SMI_DEFAULT_STYLE PROPERTY STRINGS plain pretty)
if(ECLAIR_SMI_DEFAULT_STYLE STREQUAL "pretty")
    set(eclair_smi_default_pretty 1)
elseif(ECLAIR_SMI_DEFAULT_STYLE STREQUAL "plain")
    set(eclair_smi_default_pretty 0)
else()
    message(FATAL_ERROR "ECLAIR_SMI_DEFAULT_STYLE must be plain or pretty")
endif()

add_executable(eclair-smi main.cpp ${eclair_smi_backends})
target_link_libraries(eclair-smi
    PRIVATE eclair_smi_core eclair_backend_host ${eclair_smi_backend_libraries})
target_compile_definitions(eclair-smi PRIVATE
    ECLAIR_VERSION_STRING="${PROJECT_VERSION}"
    ECLAIR_SMI_DEFAULT_PRETTY=${eclair_smi_default_pretty})
# Next to the other diagnostics at the top of the build tree.
set_target_properties(eclair-smi PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# Keep the toolkit's library directory after install, so the installed tool
# finds the same CUDA runtime it was linked against.
set_target_properties(eclair-smi PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
eclair_target_warnings(eclair-smi)
install(TARGETS eclair-smi RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(ECLAIR_BUILD_TESTS)
    add_executable(eclair_smi_tests ${PROJECT_SOURCE_DIR}/tests/eclair_smi_tests.cpp)
    target_link_libraries(eclair_smi_tests
        PRIVATE eclair_smi_core eclair_backend_host eclair_cuda_architecture)
    eclair_target_warnings(eclair_smi_tests)
    add_test(NAME eclair.smi COMMAND eclair_smi_tests)

    # End-to-end runs of the real binary. On a CUDA build these probe the
    # GPU through the runtime API, as the tool itself does.
    add_test(NAME eclair.smi_report COMMAND eclair-smi --color=never)
    add_test(NAME eclair.smi_json COMMAND eclair-smi --json)
    set_tests_properties(eclair.smi_json PROPERTIES
        PASS_REGULAR_EXPRESSION "\"schema\": \"eclair-smi.report/2\"")
    add_test(NAME eclair.smi_usage_error COMMAND eclair-smi --no-such-option)
    add_test(NAME eclair.smi_policy_error COMMAND eclair-smi)
    add_test(NAME eclair.smi_style_error COMMAND eclair-smi)
    set_tests_properties(eclair.smi_style_error PROPERTIES
        WILL_FAIL TRUE ENVIRONMENT "ECLAIR_SMI_STYLE=sparkly")
    # Pretty output to a pipe stays free of escape codes unless forced.
    add_test(NAME eclair.smi_pretty_pipe COMMAND eclair-smi --pretty)
    string(ASCII 27 eclair_escape)
    set_tests_properties(eclair.smi_pretty_pipe PROPERTIES
        FAIL_REGULAR_EXPRESSION "${eclair_escape}\\[")
    set_tests_properties(eclair.smi_usage_error eclair.smi_policy_error PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(eclair.smi_policy_error PROPERTIES ENVIRONMENT "ECLAIR_DISABLE=fp9.*")
endif()
