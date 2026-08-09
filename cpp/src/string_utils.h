#pragma once

#include <string>

std::string decodeBase64(const std::string& encodedString);
bool parsePositiveInteger(const char* value, int* result);
void replaceSubstring(std::string& value, const std::string& from, const std::string& to);