cmake_minimum_required(VERSION 3.14)
project(ofxStableDiffusionTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_executable(test_video_helpers
    test_video_helpers.cpp
)

target_include_directories(test_video_helpers PRIVATE
    "${SRC_DIR}"
)

add_test(NAME video_helpers COMMAND test_video_helpers)

add_executable(test_image_helpers
    test_image_helpers.cpp
)

target_include_directories(test_image_helpers PRIVATE
    "${SRC_DIR}"
)

add_test(NAME image_helpers COMMAND test_image_helpers)

add_executable(test_ranking_helpers
    test_ranking_helpers.cpp
)

target_include_directories(test_ranking_helpers PRIVATE
    "${SRC_DIR}"
)

add_test(NAME ranking_helpers COMMAND test_ranking_helpers)

add_executable(test_image_helpers_extended
    test_image_helpers_extended.cpp
)

target_include_directories(test_image_helpers_extended PRIVATE
    "${SRC_DIR}"
)

add_test(NAME image_helpers_extended COMMAND test_image_helpers_extended)

add_executable(test_ranking_helpers_extended
    test_ranking_helpers_extended.cpp
)

target_include_directories(test_ranking_helpers_extended PRIVATE
    "${SRC_DIR}"
)

add_test(NAME ranking_helpers_extended COMMAND test_ranking_helpers_extended)

add_executable(test_video_helpers_extended
    test_video_helpers_extended.cpp
)

target_include_directories(test_video_helpers_extended PRIVATE
    "${SRC_DIR}"
)

add_test(NAME video_helpers_extended COMMAND test_video_helpers_extended)
