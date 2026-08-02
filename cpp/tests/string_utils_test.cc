#include <string>

#include <gtest/gtest.h>

#include "string_utils.h"

namespace {

TEST(StringUtilsTest, ReplaceSubstringReplacesSingleOccurrence)
{
    std::string value = "hello world";

    replaceSubstring(value, "world", "there");

    EXPECT_EQ(value, "hello there");
}

TEST(StringUtilsTest, ReplaceSubstringReplacesEveryOccurrence)
{
    std::string value = "one fish, two fish, red fish";

    replaceSubstring(value, "fish", "cat");

    EXPECT_EQ(value, "one cat, two cat, red cat");
}

TEST(StringUtilsTest, ReplaceSubstringSupportsLongerReplacement)
{
    std::string value = "aaa";

    replaceSubstring(value, "a", "xyz");

    EXPECT_EQ(value, "xyzxyzxyz");
}

TEST(StringUtilsTest, ReplaceSubstringSupportsDeletion)
{
    std::string value = "remove this this text";

    replaceSubstring(value, "this ", "");

    EXPECT_EQ(value, "remove text");
}

TEST(StringUtilsTest, ReplaceSubstringLeavesValueUnchangedWhenSearchIsMissing)
{
    std::string value = "hello";

    replaceSubstring(value, "world", "there");

    EXPECT_EQ(value, "hello");
}

TEST(StringUtilsTest, ReplaceSubstringLeavesValueUnchangedForEmptySearch)
{
    std::string value = "hello";

    replaceSubstring(value, "", "there");

    EXPECT_EQ(value, "hello");
}

TEST(StringUtilsTest, DecodeBase64HandlesStandardPaddingLengths)
{
    EXPECT_EQ(decodeBase64("Zg=="), "f");
    EXPECT_EQ(decodeBase64("Zm8="), "fo");
    EXPECT_EQ(decodeBase64("Zm9v"), "foo");
}

TEST(StringUtilsTest, DecodeBase64HandlesMultipleBlocks)
{
    EXPECT_EQ(decodeBase64("SGVsbG8sIHdvcmxkIQ=="), "Hello, world!");
}

TEST(StringUtilsTest, DecodeBase64PreservesEmbeddedNullBytes)
{
    const std::string expected("a\0b", 3);

    EXPECT_EQ(decodeBase64("YQBi"), expected);
}

TEST(StringUtilsTest, DecodeBase64ReturnsSpaceForLeadingPadding)
{
    EXPECT_EQ(decodeBase64("===="), " ");
}

TEST(StringUtilsDeathTest, DecodeBase64ExitsForEmptyInput)
{
    EXPECT_EXIT(decodeBase64(""), testing::ExitedWithCode(255), "Cannot decode an empty");
}

TEST(StringUtilsDeathTest, DecodeBase64ExitsForInvalidCharacter)
{
    EXPECT_EXIT(decodeBase64("Zm$v"), testing::ExitedWithCode(255), "Unknown character");
}

}  // namespace
