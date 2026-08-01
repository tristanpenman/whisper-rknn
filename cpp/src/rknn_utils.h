#pragma once

#include <string>

#include <rknn_api.h>

namespace rknn_utils {

const char* rknnErrorMessage(int ret);
std::string tensorAttrToString(const rknn_tensor_attr& attr);
void logRknnVersion(rknn_context ctx);
void logRkllmVersion();

}  // namespace rknn_utils
