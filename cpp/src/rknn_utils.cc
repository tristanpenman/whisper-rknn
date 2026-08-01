#include "rknn_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <rknn_api.h>

#include "logger.h"

namespace {

std::optional<std::string> findLoadedLibraryPath(std::string_view libraryName)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(libraryName) == std::string::npos) {
            continue;
        }

        const auto pathStart = line.find('/');
        if (pathStart == std::string::npos) {
            continue;
        }

        return line.substr(pathStart);
    }

    return std::nullopt;
}

std::optional<std::string> readEmbeddedVersionString(
    const std::string& libraryPath,
    std::string_view marker)
{
    std::ifstream library(libraryPath, std::ios::binary);
    if (!library) {
        return std::nullopt;
    }

    std::string carry;
    std::vector<char> buffer(64 * 1024);
    while (library) {
        library.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto bytesRead = library.gcount();
        if (bytesRead <= 0) {
            break;
        }

        std::string chunk = carry;
        chunk.append(buffer.data(), static_cast<std::size_t>(bytesRead));
        const auto markerPos = chunk.find(marker);
        if (markerPos != std::string::npos) {
            const auto valueStart = markerPos + marker.size();
            auto valueEnd = valueStart;
            const auto valueLimit = std::min(chunk.size(), valueStart + 160);
            while (valueEnd < valueLimit && chunk[valueEnd] != '\0' && chunk[valueEnd] != '\n') {
                ++valueEnd;
            }
            return std::string(marker) + chunk.substr(valueStart, valueEnd - valueStart);
        }

        const auto carrySize = std::min(chunk.size(), marker.size() + 160);
        carry = chunk.substr(chunk.size() - carrySize);
    }

    return std::nullopt;
}

}  // namespace

namespace rknn_utils {

const char* rknnErrorMessage(int ret)
{
    switch (ret) {
    case RKNN_SUCC:
        return "RKNN_SUCC (0): execute succeeded";
    case RKNN_ERR_FAIL:
        return "RKNN_ERR_FAIL (-1): execute failed";
    case RKNN_ERR_TIMEOUT:
        return "RKNN_ERR_TIMEOUT (-2): execute timed out";
    case RKNN_ERR_DEVICE_UNAVAILABLE:
        return "RKNN_ERR_DEVICE_UNAVAILABLE (-3): device is unavailable";
    case RKNN_ERR_MALLOC_FAIL:
        return "RKNN_ERR_MALLOC_FAIL (-4): memory allocation failed";
    case RKNN_ERR_PARAM_INVALID:
        return "RKNN_ERR_PARAM_INVALID (-5): parameter is invalid";
    case RKNN_ERR_MODEL_INVALID:
        return "RKNN_ERR_MODEL_INVALID (-6): model is invalid";
    case RKNN_ERR_CTX_INVALID:
        return "RKNN_ERR_CTX_INVALID (-7): context is invalid";
    case RKNN_ERR_INPUT_INVALID:
        return "RKNN_ERR_INPUT_INVALID (-8): input is invalid";
    case RKNN_ERR_OUTPUT_INVALID:
        return "RKNN_ERR_OUTPUT_INVALID (-9): output is invalid";
    case RKNN_ERR_DEVICE_UNMATCH:
        return "RKNN_ERR_DEVICE_UNMATCH (-10): SDK and NPU driver or firmware do not match";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
        return "RKNN_ERR_INCOMPATIBLE_PRE_COMPILE_MODEL (-11): pre-compiled model is incompatible";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
        return "RKNN_ERR_INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION (-12): optimization level is incompatible";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
        return "RKNN_ERR_TARGET_PLATFORM_UNMATCH (-13): model target platform does not match current platform";
    default:
        return "Unknown RKNN error code";
    }
}

std::string tensorAttrToString(const rknn_tensor_attr& attr)
{
    std::ostringstream stream;
    stream << "index=" << attr.index
           << " name=" << attr.name
           << " n_dims=" << attr.n_dims
           << " dims=[";
    for (std::uint32_t i = 0; i < attr.n_dims && i < RKNN_MAX_DIMS; ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << attr.dims[i];
    }
    stream << "] n_elems=" << attr.n_elems
           << " size=" << attr.size
           << " fmt=" << get_format_string(attr.fmt)
           << " type=" << get_type_string(attr.type)
           << " qnt_type=" << get_qnt_type_string(attr.qnt_type)
           << " zp=" << attr.zp
           << " scale=" << attr.scale;
    return stream.str();
}

void logRknnVersion(rknn_context ctx)
{
    rknn_sdk_version rknnVersion;
    memset(&rknnVersion, 0, sizeof(rknnVersion));
    const int ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &rknnVersion, sizeof(rknnVersion));
    if (ret == RKNN_SUCC) {
        LOG(INFO) << "RKNN version: api=" << rknnVersion.api_version
                  << " driver=" << rknnVersion.drv_version;
    } else {
        LOG(WARNING) << "Failed to query RKNN version, error=" << rknnErrorMessage(ret);
    }
}

void logRkllmVersion()
{
    const auto libraryPath = findLoadedLibraryPath("librkllmrt.so");
    if (!libraryPath.has_value()) {
        LOG(WARNING) << "RKLLM version: unavailable (librkllmrt.so is not visible in /proc/self/maps)";
        return;
    }

    const auto version = readEmbeddedVersionString(
        *libraryPath,
        "rknn llm lib version: ");
    if (version.has_value()) {
        LOG(INFO) << "RKLLM version: " << *version;
        return;
    }

    const auto sdkVersion = readEmbeddedVersionString(
        *libraryPath,
        "RKLLM SDK (version: ");
    if (sdkVersion.has_value()) {
        LOG(INFO) << "RKLLM version: " << *sdkVersion;
        return;
    }

    LOG(WARNING) << "RKLLM version: unavailable (no embedded version marker found in "
                 << *libraryPath << ")";
}

}  // namespace rknn_utils
