# OpenCV third-party dependency configuration.
#
# This intentionally keeps the fetched OpenCV build small for the first
# integration step. Add modules to VLM_RKNN_OPENCV_MODULES as the runtime
# starts using more image/video functionality.

set(VLM_RKNN_OPENCV_GIT_TAG "4.13.0" CACHE STRING
    "OpenCV git tag, branch, or commit fetched when VLM_RKNN_ENABLE_OPENCV is ON")
set(VLM_RKNN_OPENCV_MODULES "core,imgproc,imgcodecs" CACHE STRING
    "Comma-separated OpenCV modules to build via FetchContent")

# We don't want to build OpenCV samples by default
option(VLM_RKNN_OPENCV_BUILD_SAMPLES "Build OpenCV sample projects" OFF)

# Keep OpenCV focused on embeddable C++ libraries for Rockchip Linux/Android
# targets. These cache entries are consumed by OpenCV's own CMake project.
set(BUILD_LIST "${VLM_RKNN_OPENCV_MODULES}" CACHE STRING "OpenCV modules to build" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "Build OpenCV command-line applications" FORCE)
set(BUILD_opencv_js OFF CACHE BOOL "Build OpenCV.js bindings" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "Build OpenCV Java bindings" FORCE)
set(BUILD_opencv_java OFF CACHE BOOL "Build OpenCV Java bindings" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "Build OpenCV Python 2 bindings" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "Build OpenCV Python 3 bindings" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build OpenCV tests" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "Build OpenCV performance tests" FORCE)
set(BUILD_EXAMPLES ${VLM_RKNN_OPENCV_BUILD_SAMPLES} CACHE BOOL "Build OpenCV examples" FORCE)
set(BUILD_ANDROID_EXAMPLES ${VLM_RKNN_OPENCV_BUILD_SAMPLES} CACHE BOOL "Build OpenCV Android examples" FORCE)
set(BUILD_ANDROID_PROJECTS ${VLM_RKNN_OPENCV_BUILD_SAMPLES} CACHE BOOL "Build OpenCV Android projects" FORCE)
set(INSTALL_ANDROID_EXAMPLES ${VLM_RKNN_OPENCV_BUILD_SAMPLES} CACHE BOOL "Install OpenCV Android examples" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "Build OpenCV documentation" FORCE)
set(OPENCV_ENABLE_NONFREE OFF CACHE BOOL "Enable OpenCV non-free algorithms" FORCE)

# Keep imgcodecs limited to the image formats used by this demo.
set(WITH_JPEG ON CACHE BOOL "Use JPEG image codec" FORCE)
set(WITH_PNG ON CACHE BOOL "Use PNG image codec" FORCE)
set(WITH_SPNG OFF CACHE BOOL "Use alternative SPNG image codec" FORCE)
set(WITH_WEBP OFF CACHE BOOL "Use WebP image codec" FORCE)
set(WITH_TIFF OFF CACHE BOOL "Use TIFF image codec" FORCE)
set(WITH_JASPER OFF CACHE BOOL "Use JPEG-2000 Jasper image codec" FORCE)
set(WITH_OPENJPEG OFF CACHE BOOL "Use OpenJPEG image codec" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "Use OpenEXR" FORCE)
set(WITH_JPEGXL OFF CACHE BOOL "Use JPEG XL image codec" FORCE)
set(WITH_AVIF OFF CACHE BOOL "Use AVIF image codec" FORCE)
set(WITH_GDAL OFF CACHE BOOL "Use GDAL image I/O" FORCE)
set(WITH_GDCM OFF CACHE BOOL "Use GDCM DICOM image I/O" FORCE)
set(WITH_IMGCODEC_GIF OFF CACHE BOOL "Enable GIF image I/O" FORCE)
set(WITH_IMGCODEC_HDR OFF CACHE BOOL "Enable Radiance HDR image I/O" FORCE)
set(WITH_IMGCODEC_PXM OFF CACHE BOOL "Enable PNM/PBM/PGM/PPM image I/O" FORCE)
set(WITH_IMGCODEC_PFM OFF CACHE BOOL "Enable PFM image I/O" FORCE)
set(WITH_IMGCODEC_SUNRASTER OFF CACHE BOOL "Enable Sun raster image I/O" FORCE)
set(BUILD_JPEG ON CACHE BOOL "Build bundled JPEG codec" FORCE)
set(BUILD_PNG ON CACHE BOOL "Build bundled PNG codec" FORCE)
set(BUILD_WEBP OFF CACHE BOOL "Build bundled WebP codec" FORCE)
set(BUILD_TIFF OFF CACHE BOOL "Build bundled TIFF codec" FORCE)
set(BUILD_JASPER OFF CACHE BOOL "Build bundled Jasper codec" FORCE)
set(BUILD_OPENJPEG OFF CACHE BOOL "Build bundled OpenJPEG codec" FORCE)
set(BUILD_OPENEXR OFF CACHE BOOL "Build bundled OpenEXR codec" FORCE)
set(BUILD_AVIF OFF CACHE BOOL "Build bundled AVIF codec" FORCE)
set(OPENCV_IO_ENABLE_JASPER OFF CACHE BOOL "Enable JPEG-2000 Jasper image I/O" FORCE)
set(OPENCV_IO_ENABLE_OPENEXR OFF CACHE BOOL "Enable OpenEXR image I/O" FORCE)
set(OPENCV_IO_ENABLE_HDR OFF CACHE BOOL "Enable Radiance HDR image I/O" FORCE)
set(OPENCV_IO_ENABLE_PXM OFF CACHE BOOL "Enable PNM/PBM/PGM/PPM image I/O" FORCE)
set(OPENCV_IO_ENABLE_PFM OFF CACHE BOOL "Enable PFM image I/O" FORCE)
set(OPENCV_IO_ENABLE_SUNRASTER OFF CACHE BOOL "Enable Sun raster image I/O" FORCE)

# Avoid probing or linking optional desktop/media stacks that are not needed for
# the initial embedded image-preprocessing dependency.
set(WITH_FFMPEG OFF CACHE BOOL "Use FFmpeg" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "Use GStreamer" FORCE)
set(WITH_GTK OFF CACHE BOOL "Use GTK" FORCE)
set(WITH_QT OFF CACHE BOOL "Use Qt" FORCE)
set(WITH_IPP OFF CACHE BOOL "Use Intel IPP" FORCE)
set(WITH_ITT OFF CACHE BOOL "Use Intel ITT tracing" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "Use OpenCL" FORCE)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG ${WHISPER_RKNN_OPENCV_GIT_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(opencv)

# OpenCV's in-tree consumption (via add_subdirectory) does not set
# INTERFACE_INCLUDE_DIRECTORIES on the module targets, so downstream consumers
# need the public header directories wired up explicitly.
set(WHISPER_RKNN_OPENCV_INCLUDE_DIRS
    "${opencv_SOURCE_DIR}/include"
    "${opencv_SOURCE_DIR}/modules/core/include"
    "${opencv_SOURCE_DIR}/modules/imgproc/include"
    "${opencv_SOURCE_DIR}/modules/imgcodecs/include"
    "${CMAKE_BINARY_DIR}"
)

set(WHISPER_RKNN_OPENCV_TARGETS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)
