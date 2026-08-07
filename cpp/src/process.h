#pragma once

#include <string>
#include <vector>

#include "audio_utils.h"

inline constexpr int kVocabSize = 51865;
inline constexpr int kDefaultMaxTokens = 12;
inline constexpr int kSampleRate = 16000;
inline constexpr int kFftSize = 400;
inline constexpr int kHopLength = 160;
inline constexpr int kDefaultChunkLength = 20;
inline constexpr int kNumMels = 80;
inline constexpr int kMelFilterSize = 201;
inline constexpr char kMelFiltersPath[] = "./model/mel_80_filters.txt";

struct VocabEntry
{
    int index = 0;
    std::string token;
};

int readVocab(const char* fileName, VocabEntry* vocab);

int readMelFilters(const char* fileName, float* data, int maxLines);

//
// Applies endpoint-excluding reflection padding for ordinary inputs
//
// For example, given an input of [0, 1, 2] and a padding width of 2,
// the output will be [2, 1, 0, 1, 2, 1, 0].
//
// Like PyTorch, padding must be strictly smaller than the input size. These
// edge cases throw std::invalid_argument:
// - padding equal to the input size: [0, 1, 2] padded by 3
// - padding wider than the input: [0, 1, 2] padded by 4
// - padded single-sample input: [3] padded by 1
// - empty input: [] padded by 0
//
// A padding width of 0 is valid for any non-empty input and copies it unchanged.
//
void reflectPad(
    const std::vector<float>& input,
    std::vector<float>& output,
    int paddingWidth);

void preprocessAudio(
    AudioBuffer* audio,
    const float* melFilters,
    std::vector<float>& melSpectrogram,
    int chunkLength,
    bool enableNeon = true);

void multiplyMatrices(
    const float* left,
    const float* right,
    std::vector<float>& output,
    int leftRows,
    int sharedColumns,
    int rightColumns,
    bool enableNeon);

int argmax(const float* array, int tokenIndex);
