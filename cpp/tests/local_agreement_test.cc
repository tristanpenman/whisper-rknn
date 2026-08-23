#include <vector>

#include <gtest/gtest.h>

#include "local_agreement.h"

namespace {

TEST(LocalAgreementTest, FirstHypothesisIsProvisional)
{
    LocalAgreement agreement;

    const LocalAgreementUpdate result = agreement.update({1, 2, 3});

    EXPECT_TRUE(result.committedTokens.empty());
    EXPECT_EQ(result.provisionalTokens, (std::vector<int>{1, 2, 3}));
    EXPECT_FALSE(result.isFinal);
}

TEST(LocalAgreementTest, CommitsCommonPrefixAndRetainsLatestSuffix)
{
    LocalAgreement agreement;
    agreement.update({1, 2, 3});

    const LocalAgreementUpdate result = agreement.update({1, 2, 4, 5});

    EXPECT_EQ(result.committedTokens, (std::vector<int>{1, 2}));
    EXPECT_EQ(result.provisionalTokens, (std::vector<int>{4, 5}));
    EXPECT_EQ(agreement.committedTokens(), (std::vector<int>{1, 2}));
}

TEST(LocalAgreementTest, CommitsPreviouslyProvisionalTokensOnlyOnce)
{
    LocalAgreement agreement;
    agreement.update({1, 2, 3});
    agreement.update({1, 2, 4});

    const LocalAgreementUpdate result = agreement.update({4, 5});

    EXPECT_EQ(result.committedTokens, (std::vector<int>{4}));
    EXPECT_EQ(result.provisionalTokens, (std::vector<int>{5}));
    EXPECT_EQ(agreement.committedTokens(), (std::vector<int>{1, 2, 4}));
}

TEST(LocalAgreementTest, FinishCommitsLatestProvisionalHypothesis)
{
    LocalAgreement agreement;
    agreement.update({7, 8});

    const LocalAgreementUpdate result = agreement.finish();

    EXPECT_EQ(result.committedTokens, (std::vector<int>{7, 8}));
    EXPECT_TRUE(result.provisionalTokens.empty());
    EXPECT_TRUE(result.isFinal);
}

TEST(LocalAgreementTest, ResetStartsASeparateStream)
{
    LocalAgreement agreement;
    agreement.update({1});
    agreement.finish();
    agreement.reset();

    const LocalAgreementUpdate result = agreement.update({2});

    EXPECT_TRUE(result.committedTokens.empty());
    EXPECT_EQ(result.provisionalTokens, (std::vector<int>{2}));
    EXPECT_TRUE(agreement.committedTokens().empty());
}

}  // namespace
