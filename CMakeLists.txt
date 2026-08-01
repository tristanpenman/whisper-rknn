cmake_minimum_required(VERSION 3.20)
project(whisper-rknn VERSION 0.1.0 LANGUAGES C CXX)

set(COPYRIGHT "Copyright (c) 2026 Tristan Penman")
set(IDENTIFIER "com.tristanpenman.whisper-rknn")

# Force out-of-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid language extensions to ensure consistent behavior across compilers
set(CMAKE_CXX_EXTENSIONS OFF)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Default to Debug for single-config generators if user didn't specify anything
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Address sanitizer
if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# -----------------------------------------------------------------------------
# build config
# -----------------------------------------------------------------------------

option(WHISPER_RKNN_ENABLE_OPENCV "Fetch and link OpenCV as a third-party dependency" ON)

if(ANDROID)
    set(WHISPER_RKNN_ARCH ${CMAKE_ANDROID_ARCH_ABI})
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(WHISPER_RKNN_ARCH aarch64)
    else()
        set(WHISPER_RKNN_ARCH armhf)
    endif()
endif()

# -----------------------------------------------------------------------------
# OpenCV
# -----------------------------------------------------------------------------

if(WHISPER_RKNN_ENABLE_OPENCV)
    include(cmake/OpenCV.cmake)
endif()

# -----------------------------------------------------------------------------
# rknn runtime
# -----------------------------------------------------------------------------

set(DEFAULT_RKNN_RUNTIME_LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/${CMAKE_SYSTEM_NAME}/${WHISPER_RKNN_ARCH}/librknnrt.so)
set(RKNN_RUNTIME_LIB ${DEFAULT_RKNN_RUNTIME_LIB} CACHE FILEPATH
    "Path to librknnrt.so for the selected target")
set(RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/include CACHE PATH
    "Path containing rknn_api.h")

if(NOT EXISTS ${RKNN_RUNTIME_LIB})
    message(FATAL_ERROR
        "RKNN runtime library not found: ${RKNN_RUNTIME_LIB}. "
        "Set -DRKNN_RUNTIME_LIB=<path-to-librknnrt.so> (for Android use arm64-v8a runtime).")
endif()

if(NOT EXISTS ${RKNN_INCLUDE_DIR}/rknn_api.h)
    message(FATAL_ERROR
        "RKNN headers not found in ${RKNN_INCLUDE_DIR}. "
        "Set -DRKNN_INCLUDE_DIR=<path-containing-rknn_api.h>.")
endif()

# -----------------------------------------------------------------------------
# libsndfile
# -----------------------------------------------------------------------------

set(LIBSNDFILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libsndfile)
set(LIBSNDFILE_INCLUDES ${LIBSNDFILE_PATH}/include)
set(LIBSNDFILE ${LIBSNDFILE_PATH}/${CMAKE_SYSTEM_NAME}/${WHISPER_RKNN_ARCH}/libsndfile.a)

# -----------------------------------------------------------------------------
# fftw
# -----------------------------------------------------------------------------

set(LIBFFTW_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fftw)
set(LIBFFTW_INCLUDES ${LIBFFTW_PATH}/include )
set(LIBFFTW ${LIBFFTW_PATH}/${CMAKE_SYSTEM_NAME}/${WHISPER_RKNN_ARCH}/libfftw3f.a)

# -----------------------------------------------------------------------------
# whisper_rknn
# -----------------------------------------------------------------------------

add_executable(whisper_rknn
    cpp/src/audio_utils.c
    cpp/src/file_utils.c
    cpp/src/logger.cc
    cpp/src/main.cc
    cpp/src/process.cc
    cpp/src/whisper.cc
)

target_include_directories(whisper_rknn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBFFTW_INCLUDES}
    ${LIBSNDFILE_INCLUDES}
    ${LIBTIMER_INCLUDES}
    ${RKNN_INCLUDE_DIR}
)

target_link_libraries(whisper_rknn
    ${RKNN_RUNTIME_LIB}
    ${LIBFFTW}
    ${LIBSNDFILE}
)

if(WHISPER_RKNN_ENABLE_OPENCV)
    target_include_directories(whisper_rknn PRIVATE
        ${WHISPER_RKNN_OPENCV_INCLUDE_DIRS}
    )
    target_link_libraries(whisper_rknn
        ${WHISPER_RKNN_OPENCV_TARGETS}
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(whisper_rknn
    log
)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The bundled FFTW archive was built without -fPIC. Debian's cross compiler
    # links PIE executables by default, so explicitly request a non-PIE link.
    include(CheckPIESupported)
    check_pie_supported(LANGUAGES CXX)
    set_target_properties(whisper_rknn PROPERTIES POSITION_INDEPENDENT_CODE OFF)

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(whisper_rknn Threads::Threads)
endif()
