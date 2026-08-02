#pragma once

#include <string>
#include <vector>

#include "audio_utils.h"

inline constexpr int kVocabSize = 51865;
inline constexpr int kMaxTokens = 12;
inline constexpr int kSampleRate = 16000;
inline constexpr int kFftSize = 400;
inline constexpr int kHopLength = 160;
inline constexpr int kChunkLength = 20;
inline constexpr int kMaxAudioLength = kChunkLength * kSampleRate;
inline constexpr int kNumMels = 80;
inline constexpr int kMelFilterSize = 201;
inline constexpr int kEncoderInputSize = kChunkLength * 100;
inline constexpr int kEncoderOutputSize = kChunkLength * 50 * 512;
inline constexpr int kDecoderInputSize = kEncoderOutputSize;
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
    std::vector<float>& melSpectrogram);
int argmax(const float* array);
