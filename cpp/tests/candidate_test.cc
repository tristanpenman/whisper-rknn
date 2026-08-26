#include <vector>

#include <gtest/gtest.h>

#include "candidate.h"

namespace {

constexpr int kTimestampBeginToken = 50364;

TEST(CandidateTest, TimestampedTextTokensRecordsCompletedSegmentEndFrames)
{
    const candidate::AgreementCandidate result = candidate::timestampedTextTokens(
        {kTimestampBeginToken, 10, 11, kTimestampBeginToken + 2, 12, kTimestampBeginToken + 4}, 1000, 3000, 1000);

    EXPECT_EQ(result.tokenIds, (std::vector<int>{10, 11, 12}));
    EXPECT_EQ(result.tokenEndFrames, (std::vector<int>{1640, 1640, 2280}));
}

TEST(CandidateTest, TimestampedTextTokensUsesWindowEndForTrailingSegment)
{
    const candidate::AgreementCandidate result = candidate::timestampedTextTokens(
        {kTimestampBeginToken, 20, 21}, 1000, 3000, 1000);

    EXPECT_EQ(result.tokenIds, (std::vector<int>{20, 21}));
    EXPECT_EQ(result.tokenEndFrames, (std::vector<int>{3000, 3000}));
}

TEST(CandidateTest, TextTokensFiltersInvalidControlAndTimestampTokens)
{
    EXPECT_EQ(candidate::textTokens({-1, 10, 50256, 50257, kTimestampBeginToken}), (std::vector<int>{10, 50256}));
}

TEST(CandidateTest, CommittedContextLengthIncludesTokensBeforeTheOverlap)
{
    EXPECT_EQ(candidate::committedContextLength({8, 9, 3, 4, 5}, {1, 2, 3, 4}), 4U);
    EXPECT_EQ(candidate::committedContextLength({4, 5}, {1, 2, 3, 4}), 1U);
    EXPECT_EQ(candidate::committedContextLength({5, 6}, {1, 2, 3, 4}), 0U);
}

TEST(CandidateTest, RemoveCommittedPrefixKeepsTokenEndFramesAligned)
{
    candidate::AgreementCandidate value = {{8, 9, 3, 4, 5}, {100, 100, 200, 200, 300}};

    candidate::removeCommittedPrefix(value, {1, 2, 3, 4});

    EXPECT_EQ(value.tokenIds, (std::vector<int>{5}));
    EXPECT_EQ(value.tokenEndFrames, (std::vector<int>{300}));
}

TEST(CandidateTest, CommittedSegmentEndFrameAdvancesOnlyThroughCompletedSegments)
{
    const candidate::AgreementCandidate value = {{10, 11, 12, 13}, {100, 100, 200, 200}};

    EXPECT_EQ(candidate::committedSegmentEndFrame(value, 3, 50), 100);
    EXPECT_EQ(candidate::committedSegmentEndFrame(value, 4, 50), 200);
    EXPECT_EQ(candidate::committedSegmentEndFrame(value, 4, 250), 250);
}

}  // namespace
