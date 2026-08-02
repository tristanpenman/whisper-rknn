#include "string_utils.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

std::int32_t base64CharacterIndex(char character)
{
    if (character >= 'A' && character <= 'Z') {
        return character - 'A';
    }
    if (character >= 'a' && character <= 'z') {
        return character - 'a' + ('Z' - 'A') + 1;
    }
    if (character >= '0' && character <= '9') {
        return character - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
    }
    if (character == '+') {
        return 62;
    }
    if (character == '/') {
        return 63;
    }

    std::cerr << "Unknown character " << static_cast<int>(character) << ", " << character << '\n';
    std::exit(-1);
}

}  // namespace

void replaceSubstring(
    std::string& value,
    const std::string& from,
    const std::string& to)
{
    if (from.empty()) {
        return;
    }

    std::size_t startPosition = 0;
    while ((startPosition = value.find(from, startPosition)) != std::string::npos) {
        value.replace(startPosition, from.length(), to);
        startPosition += to.length();
    }
}

std::string decodeBase64(const std::string& encodedString)
{
    if (encodedString.empty()) {
        std::cerr << "Cannot decode an empty Base64 string.\n";
        std::exit(-1);
    }

    const std::int32_t outputLength = static_cast<std::int32_t>(encodedString.size()) / 4 * 3;
    std::string decodedString;
    decodedString.reserve(outputLength);

    std::int32_t index = 0;
    while (index < static_cast<std::int32_t>(encodedString.size())) {
        if (encodedString[index] == '=') {
            return " ";
        }

        const std::int32_t firstByte =
            (base64CharacterIndex(encodedString[index]) << 2)
            + ((base64CharacterIndex(encodedString[index + 1]) & 0x30) >> 4);
        decodedString.push_back(static_cast<char>(firstByte));

        if (index + 2 < static_cast<std::int32_t>(encodedString.size())
            && encodedString[index + 2] != '=') {
            const std::int32_t secondByte =
                ((base64CharacterIndex(encodedString[index + 1]) & 0x0f) << 4)
                + ((base64CharacterIndex(encodedString[index + 2]) & 0x3c) >> 2);
            decodedString.push_back(static_cast<char>(secondByte));

            if (index + 3 < static_cast<std::int32_t>(encodedString.size())
                && encodedString[index + 3] != '=') {
                const std::int32_t thirdByte =
                    ((base64CharacterIndex(encodedString[index + 2]) & 0x03) << 6)
                    + base64CharacterIndex(encodedString[index + 3]);
                decodedString.push_back(static_cast<char>(thirdByte));
            }
        }
        index += 4;
    }

    return decodedString;
}
