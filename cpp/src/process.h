#pragma once

#include <string>
#include <vector>

#include "audio_utils.h"

inline constexpr int kVocabSize = 51865;
inline constexpr int kDefaultMaxTokens = 64;
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
