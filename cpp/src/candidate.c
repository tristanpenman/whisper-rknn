#include "candidate.h"

#include <algorithm>

#include "process.h"

namespace {

constexpr int kEndOfTextToken = 50257;
constexpr int kTimestampBeginToken = 50364;
constexpr int kTimestampFrames = kSampleRate / 50;

}  // namespace

namespace candidate {

AgreementCandidate timestampedTextTokens(const std::vector<int>& tokenIds, int firstFrame, int endFrame,
    int committedEndFrame)
{
    AgreementCandidate result;
    std::vector<int> segmentTokens;
    for (const int tokenId : tokenIds) {
        if (tokenId >= kTimestampBeginToken) {
            const int timestampFrame = std::min(
                firstFrame + (tokenId - kTimestampBeginToken) * kTimestampFrames, endFrame);
            if (!segmentTokens.empty()) {
                if (timestampFrame > committedEndFrame) {
                    result.tokenIds.insert(result.tokenIds.end(), segmentTokens.begin(), segmentTokens.end());
                    result.tokenEndFrames.insert(result.tokenEndFrames.end(), segmentTokens.size(), timestampFrame);
                }
                segmentTokens.clear();
            }
        } else if (tokenId >= 0 && tokenId < kEndOfTextToken) {
            segmentTokens.push_back(tokenId);
        }
    }

    // A trailing segment without its closing timestamp remains provisional.
    if (!segmentTokens.empty() && endFrame > committedEndFrame) {
        result.tokenIds.insert(result.tokenIds.end(), segmentTokens.begin(), segmentTokens.end());
        result.tokenEndFrames.insert(result.tokenEndFrames.end(), segmentTokens.size(), endFrame);
    }
    return result;
}

std::vector<int> textTokens(const std::vector<int>& tokenIds)
{
    std::vector<int> result;
    for (const int tokenId : tokenIds) {
        if (tokenId >= 0 && tokenId < kEndOfTextToken) {
            result.push_back(tokenId);
        }
    }
    return result;
}

std::size_t committedContextLength(const std::vector<int>& hypothesis, const std::vector<int>& committed)
{
    const std::size_t maximumLength = std::min(hypothesis.size(), committed.size());
    for (std::size_t length = maximumLength; length > 1; --length) {
        const auto match = std::find_end(
            hypothesis.begin(), hypothesis.end(), committed.end() - length, committed.end());
        if (match != hypothesis.end()) {
            return static_cast<std::size_t>(match - hypothesis.begin()) + length;
        }
    }
    if (!hypothesis.empty() && !committed.empty() && hypothesis.front() == committed.back()) {
        return 1;
    }
    return 0;
}

void removeCommittedPrefix(AgreementCandidate& candidate, const std::vector<int>& committed)
{
    const std::size_t length = committedContextLength(candidate.tokenIds, committed);
    candidate.tokenIds.erase(candidate.tokenIds.begin(), candidate.tokenIds.begin() + length);
    if (!candidate.tokenEndFrames.empty()) {
        candidate.tokenEndFrames.erase(candidate.tokenEndFrames.begin(), candidate.tokenEndFrames.begin() + length);
    }
}

int committedSegmentEndFrame(const AgreementCandidate& candidate, std::size_t committedTokenCount, int previousEndFrame)
{
    int result = previousEndFrame;
    const std::size_t limit = std::min(committedTokenCount, candidate.tokenEndFrames.size());
    for (std::size_t index = 0; index < limit; ++index) {
        const bool completesSegment = index + 1 == candidate.tokenEndFrames.size()
            || candidate.tokenEndFrames[index + 1] > candidate.tokenEndFrames[index];
        if (completesSegment) {
            result = std::max(result, candidate.tokenEndFrames[index]);
        }
    }
    return result;
}

}  // namespace candidate
