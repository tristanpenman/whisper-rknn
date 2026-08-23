#pragma once

#include <vector>

struct LocalAgreementUpdate
{
    // Tokens that became stable during this update. Emit these exactly once.
    std::vector<int> committedTokens;

    // Latest replaceable suffix. Consumers may redraw this on every update.
    std::vector<int> provisionalTokens;

    bool isFinal = false;
};

// Applies a two-hypothesis Local Agreement policy to token IDs.
//
// Hypotheses passed to update() must describe only the uncommitted audio
// region. The caller is responsible for removing control/timestamp tokens and
// for aligning a rolling-window result to the last committed timestamp.
class LocalAgreement
{
public:
    // The first hypothesis is provisional. Later calls commit the longest
    // common prefix shared with the preceding hypothesis.
    LocalAgreementUpdate update(const std::vector<int>& hypothesis);

    // Commits the latest provisional hypothesis at end-of-stream. Calling
    // finish() again returns an empty final update.
    LocalAgreementUpdate finish();

    void reset();

    const std::vector<int>& committedTokens() const;
    const std::vector<int>& provisionalTokens() const;

private:
    std::vector<int> committedTokens_;
    std::vector<int> provisionalTokens_;
    bool finished_ = false;
};
