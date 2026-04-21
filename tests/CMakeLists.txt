cmake_minimum_required(VERSION 3.14)
project(ofxStableDiffusionTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(STUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stubs")
set(SD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs/stable-diffusion/include")

add_executable(test_video_helpers
    test_video_helpers.cpp
)

target_include_directories(test_video_helpers PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME video_helpers COMMAND test_video_helpers)

add_executable(test_image_helpers
    test_image_helpers.cpp
)

target_include_directories(test_image_helpers PRIVATE
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME image_helpers COMMAND test_image_helpers)

add_executable(test_ranking_helpers
    test_ranking_helpers.cpp
)

target_include_directories(test_ranking_helpers PRIVATE
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME ranking_helpers COMMAND test_ranking_helpers)

add_executable(test_image_helpers_extended
    test_image_helpers_extended.cpp
)

target_include_directories(test_image_helpers_extended PRIVATE
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME image_helpers_extended COMMAND test_image_helpers_extended)

add_executable(test_ranking_helpers_extended
    test_ranking_helpers_extended.cpp
)

target_include_directories(test_ranking_helpers_extended PRIVATE
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME ranking_helpers_extended COMMAND test_ranking_helpers_extended)

add_executable(test_video_helpers_extended
    test_video_helpers_extended.cpp
)

target_include_directories(test_video_helpers_extended PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME video_helpers_extended COMMAND test_video_helpers_extended)

add_executable(test_capability_helpers
    test_capability_helpers.cpp
)

target_include_directories(test_capability_helpers PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME capability_helpers COMMAND test_capability_helpers)

add_executable(test_parameter_tuning_helpers
    test_parameter_tuning_helpers.cpp
)

target_include_directories(test_parameter_tuning_helpers PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME parameter_tuning_helpers COMMAND test_parameter_tuning_helpers)

add_executable(test_long_video_workflow
    test_long_video_workflow.cpp
)

target_include_directories(test_long_video_workflow PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME long_video_workflow COMMAND test_long_video_workflow)

add_executable(test_lora_discovery
    test_lora_discovery.cpp
)

target_include_directories(test_lora_discovery PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME lora_discovery COMMAND test_lora_discovery)

add_executable(test_inpainting_validation
    test_inpainting_validation.cpp
)

target_include_directories(test_inpainting_validation PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME inpainting_validation COMMAND test_inpainting_validation)

add_executable(test_model_manager_integration
    test_model_manager_integration.cpp
)

target_include_directories(test_model_manager_integration PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME model_manager_integration COMMAND test_model_manager_integration)

add_executable(test_multi_controlnet
    test_multi_controlnet.cpp
)

target_include_directories(test_multi_controlnet PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME multi_controlnet COMMAND test_multi_controlnet)

add_executable(test_performance_profiler
    test_performance_profiler.cpp
)

target_include_directories(test_performance_profiler PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME performance_profiler COMMAND test_performance_profiler)

add_executable(test_sampling_helpers
    test_sampling_helpers.cpp
)

target_include_directories(test_sampling_helpers PRIVATE
    "${STUB_DIR}"
    "${SRC_DIR}"
    "${SD_INCLUDE_DIR}"
)

add_test(NAME sampling_helpers COMMAND test_sampling_helpers)

add_executable(test_realtime_session
    test_realtime_session.cpp
)

add_test(NAME realtime_session COMMAND test_realtime_session)
