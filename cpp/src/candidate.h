#pragma once

#include <cstddef>
#include <vector>

namespace candidate {

struct AgreementCandidate
{
    std::vector<int> tokenIds;
    std::vector<int> tokenEndFrames;
};

// Extracts text tokens and their segment end frames from a timestamped hypothesis.
AgreementCandidate timestampedTextTokens(const std::vector<int>& tokenIds, int firstFrame, int endFrame,
    int committedEndFrame);

// Filters a hypothesis down to valid text tokens.
std::vector<int> textTokens(const std::vector<int>& tokenIds);

// Finds the hypothesis prefix that overlaps the end of the committed tokens.
std::size_t committedContextLength(const std::vector<int>& hypothesis, const std::vector<int>& committed);

// Removes tokens already represented by the committed context from a candidate.
void removeCommittedPrefix(AgreementCandidate& candidate, const std::vector<int>& committed);

// Advances the committed audio position to the end of the last completed segment.
int committedSegmentEndFrame(
    const AgreementCandidate& candidate, std::size_t committedTokenCount, int previousEndFrame);

}  // namespace candidate
