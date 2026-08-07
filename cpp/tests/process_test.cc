#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "process.h"

namespace {

TEST(ProcessTest, ReflectionPaddingExcludesEndpoints)
{
    const std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> output(input.size() + 4);

    reflectPad(input, output, 2);

    const std::vector<float> expected = {
        2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
    };
    EXPECT_EQ(output, expected);
}

TEST(ProcessTest, ReflectionPaddingRejectsPaddingEqualToInputSize)
{
    const std::vector<float> input = {0.0f, 1.0f, 2.0f};
    std::vector<float> output;

    EXPECT_THROW(reflectPad(input, output, 3), std::invalid_argument);
}

TEST(ProcessTest, ReflectionPaddingRejectsPaddingWiderThanInput)
{
    const std::vector<float> input = {0.0f, 1.0f, 2.0f};
    std::vector<float> output;

    EXPECT_THROW(reflectPad(input, output, 4), std::invalid_argument);
}

TEST(ProcessTest, ReflectionPaddingRejectsPaddedSingleSampleInput)
{
    const std::vector<float> input = {3.0f};
    std::vector<float> output;

    EXPECT_THROW(reflectPad(input, output, 1), std::invalid_argument);
}

TEST(ProcessTest, ReflectionPaddingRejectsEmptyInput)
{
    const std::vector<float> input;
    std::vector<float> output;

    EXPECT_THROW(reflectPad(input, output, 0), std::invalid_argument);
}

TEST(ProcessTest, ZeroWidthReflectionPaddingCopiesNonEmptyInput)
{
    const std::vector<float> input = {3.0f};
    std::vector<float> output;

    reflectPad(input, output, 0);

    EXPECT_EQ(output, input);
}

#if ENABLE_NEON
TEST(ProcessTest, NeonMatrixMultiplicationMatchesScalarImplementation)
{
    constexpr int kLeftRows = 3;
    constexpr int kSharedColumns = 5;
    constexpr int kRightColumns = 7;
    const std::vector<float> left = {
        1.0f, 0.0f, -2.0f, 3.5f, 0.25f,
        0.0f, 4.0f, 1.5f, 0.0f, -1.0f,
        -0.5f, 2.0f, 0.0f, 1.0f, 3.0f,
    };
    const std::vector<float> right = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        -1.0f, 0.5f, 2.0f, -2.0f, 1.0f, 3.0f, 0.0f,
        4.0f, -3.0f, 1.0f, 0.0f, 2.0f, -1.0f, 5.0f,
        0.25f, 1.25f, -0.75f, 2.5f, -4.0f, 0.5f, 3.0f,
        2.0f, 0.0f, -1.0f, 1.0f, 0.5f, -2.0f, 4.0f,
    };
    std::vector<float> scalar(kLeftRows * kRightColumns);
    std::vector<float> neon(kLeftRows * kRightColumns);

    multiplyMatrices(
        left.data(), right.data(), scalar,
        kLeftRows, kSharedColumns, kRightColumns, false);
    multiplyMatrices(
        left.data(), right.data(), neon,
        kLeftRows, kSharedColumns, kRightColumns, true);

    for (std::size_t index = 0; index < scalar.size(); ++index) {
        EXPECT_NEAR(neon[index], scalar[index], 1e-5f) << "output index " << index;
    }
}
#endif

}  // namespace
