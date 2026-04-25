cmake_minimum_required(VERSION 3.15)
project(ofxStableDiffusion_Integration_Tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_PROFILING "Enable performance profiling in tests" OFF)
set(MODEL_PATH "" CACHE STRING "Path to SD model file for integration tests")

# Check if we should skip integration tests
if(DEFINED ENV{INTEGRATION_TEST_SKIP})
    message(STATUS "Integration tests skipped (INTEGRATION_TEST_SKIP is set)")
    return()
endif()

# Model path from environment or cache
if(NOT MODEL_PATH)
    if(DEFINED ENV{SD_MODEL_PATH})
        set(MODEL_PATH $ENV{SD_MODEL_PATH})
    endif()
endif()

if(MODEL_PATH)
    message(STATUS "Integration tests will use model: ${MODEL_PATH}")
    add_compile_definitions(SD_MODEL_PATH="${MODEL_PATH}")
else()
    message(WARNING "No model path set - integration tests will be skipped at runtime")
endif()

# Find openFrameworks (optional, for full integration)
# If not found, tests will use minimal addon API
find_path(OF_ROOT
    NAMES libs/openFrameworks
    PATHS
        ../../../..
        $ENV{OF_ROOT}
    DOC "openFrameworks root directory"
)

if(OF_ROOT)
    message(STATUS "Found openFrameworks at: ${OF_ROOT}")
    set(OF_INCLUDE_DIR ${OF_ROOT}/libs/openFrameworks)
else()
    message(STATUS "openFrameworks not found - using standalone mode")
endif()

# Addon sources
set(ADDON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(
    ${ADDON_ROOT}/src
    ${ADDON_ROOT}/src/core
    ${ADDON_ROOT}/src/video
    ${ADDON_ROOT}/libs/stable-diffusion/include
)

if(OF_ROOT)
    include_directories(${OF_INCLUDE_DIR})
endif()

# Common test utilities
add_library(integration_test_utils STATIC
    test_utils.cpp
)

# Integration tests
enable_testing()

# Basic generation test
add_executable(test_basic_generation
    test_basic_generation.cpp
)
target_link_libraries(test_basic_generation integration_test_utils)
add_test(NAME basic_generation COMMAND test_basic_generation)

# Cancellation test
add_executable(test_cancellation
    test_cancellation.cpp
)
target_link_libraries(test_cancellation integration_test_utils)
add_test(NAME cancellation COMMAND test_cancellation)

# Image-to-image test
add_executable(test_img2img
    test_img2img.cpp
)
target_link_libraries(test_img2img integration_test_utils)
add_test(NAME img2img COMMAND test_img2img)

# Set test properties
set_tests_properties(
    basic_generation
    cancellation
    img2img
    PROPERTIES
        TIMEOUT 300
        LABELS "integration"
)

# Performance profiling
if(ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING=1)
    message(STATUS "Performance profiling enabled")
endif()

# Platform-specific settings
if(UNIX AND NOT APPLE)
    # Linux
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
elseif(APPLE)
    # macOS
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
elseif(WIN32)
    # Windows
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

message(STATUS "Integration tests configured")
message(STATUS "  Model path: ${MODEL_PATH}")
message(STATUS "  Profiling: ${ENABLE_PROFILING}")
message(STATUS "  openFrameworks: ${OF_ROOT}")
