#include "local_agreement.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace {

std::size_t commonPrefixLength(
    const std::vector<int>& previous,
    const std::vector<int>& current)
{
    const std::size_t maximumLength = std::min(previous.size(), current.size());
    std::size_t length = 0;
    while (length < maximumLength && previous[length] == current[length]) {
        ++length;
    }
    return length;
}

}  // namespace

LocalAgreementUpdate LocalAgreement::update(const std::vector<int>& hypothesis)
{
    if (finished_) {
        return {{}, provisionalTokens_, true};
    }

    const std::size_t stableLength = commonPrefixLength(provisionalTokens_, hypothesis);
    std::vector<int> newlyCommitted(
        hypothesis.begin(), hypothesis.begin() + stableLength);
    committedTokens_.insert(
        committedTokens_.end(), newlyCommitted.begin(), newlyCommitted.end());
    provisionalTokens_.assign(hypothesis.begin() + stableLength, hypothesis.end());

    return {std::move(newlyCommitted), provisionalTokens_, false};
}

LocalAgreementUpdate LocalAgreement::finish()
{
    if (finished_) {
        return {{}, {}, true};
    }

    finished_ = true;
    std::vector<int> newlyCommitted = std::move(provisionalTokens_);
    provisionalTokens_.clear();
    committedTokens_.insert(
        committedTokens_.end(), newlyCommitted.begin(), newlyCommitted.end());
    return {std::move(newlyCommitted), {}, true};
}

void LocalAgreement::reset()
{
    committedTokens_.clear();
    provisionalTokens_.clear();
    finished_ = false;
}

const std::vector<int>& LocalAgreement::committedTokens() const
{
    return committedTokens_;
}

const std::vector<int>& LocalAgreement::provisionalTokens() const
{
    return provisionalTokens_;
}
