// Copyright (c) 2026 by Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "audio_utils.h"
#include "process.h"
#include "string_utils.h"
#include "whisper.h"

namespace {

constexpr char kEnglishVocabPath[] = "./model/vocab_en.txt";
constexpr char kChineseVocabPath[] = "./model/vocab_zh.txt";
constexpr int kEnglishTaskCode = 50259;
constexpr int kChineseTaskCode = 50260;
constexpr int kUpdateLengthSeconds = 5;

void logHypothesis(
    int iteration,
    int firstFrame,
    int endFrame,
    const TranscriptionHypothesis& hypothesis)
{
    std::printf(
        "\nStreaming iteration %d (audio %.1f-%.1f seconds)\n",
        iteration,
        firstFrame / static_cast<double>(kSampleRate),
        endFrame / static_cast<double>(kSampleRate));
    std::cout << "Whisper output token IDs: ";
    for (const int tokenId : hypothesis.tokenIds) {
        std::cout << tokenId << ' ';
    }
    std::cout << "\nWhisper output text: " << hypothesis.text << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    bool enableNeon = true;
    bool enableTimestamps = false;
    int chunkLength = kDefaultChunkLength;
    int argumentOffset = 0;
    while (argumentOffset + 1 < argc) {
        const char* argument = argv[argumentOffset + 1];
        if (std::strcmp(argument, "--disable-neon") == 0) {
            enableNeon = false;
        } else if (std::strcmp(argument, "--enable-timestamps") == 0) {
            enableTimestamps = true;
        } else if (std::strcmp(argument, "--chunk-length") == 0) {
            if (argumentOffset + 2 >= argc) {
                std::printf("Missing value for %s\n", argument);
                return -1;
            }
            if (!parsePositiveInteger(argv[argumentOffset + 2], &chunkLength)) {
                std::printf("Invalid value for %s: %s\n", argument, argv[argumentOffset + 2]);
                return -1;
            }
            ++argumentOffset;
        } else {
            break;
        }
        ++argumentOffset;
    }

    if (chunkLength < kUpdateLengthSeconds) {
        std::printf(
            "--chunk-length must be at least %d\n",
            kUpdateLengthSeconds);
        return -1;
    }
    if (argc != 5 + argumentOffset) {
        std::printf(
            "%s [--disable-neon] [--enable-timestamps] "
            "[--chunk-length <seconds>] "
            "<encoder_path> <decoder_path> <task> <audio_path>\n",
            argv[0]);
        return -1;
    }

    const char* encoderPath = argv[1 + argumentOffset];
    const char* decoderPath = argv[2 + argumentOffset];
    const char* task = argv[3 + argumentOffset];
    const char* audioPath = argv[4 + argumentOffset];
    const char* vocabPath = nullptr;
    int taskCode = 0;
    if (std::strcmp(task, "en") == 0) {
        vocabPath = kEnglishVocabPath;
        taskCode = kEnglishTaskCode;
    } else if (std::strcmp(task, "zh") == 0) {
        vocabPath = kChineseVocabPath;
        taskCode = kChineseTaskCode;
    } else {
        std::printf("Currently only English and Chinese tasks are supported.\n");
        return -1;
    }

    int result = 0;
    bool encoderInitialized = false;
    bool decoderInitialized = false;
    AudioBuffer inputAudio;
    RknnWhisperContext appContext;
    std::vector<float> melFilters(kNumMels * kMelFilterSize);
    std::vector<VocabEntry> vocab(kVocabSize);

    result = readAudio(audioPath, &inputAudio);
    if (result != 0) {
        std::printf("Failed to read audio: result=%d, path=%s\n", result, audioPath);
        goto cleanup;
    }
    if (inputAudio.numChannels == 2 && (result = convertChannels(&inputAudio)) != 0) {
        std::printf("Failed to convert audio channels: result=%d\n", result);
        goto cleanup;
    }
    if (inputAudio.sampleRate != kSampleRate
        && (result = resampleAudio(&inputAudio, inputAudio.sampleRate, kSampleRate)) != 0) {
        std::printf("Failed to resample audio: result=%d\n", result);
        goto cleanup;
    }
    if ((result = readMelFilters(kMelFiltersPath, melFilters.data(), melFilters.size())) != 0
        || (result = readVocab(vocabPath, vocab.data())) != 0) {
        std::printf("Failed to load preprocessing data: result=%d\n", result);
        goto cleanup;
    }
    if ((result = initializeWhisperModel(encoderPath, &appContext.encoderContext)) != 0) {
        std::printf("Failed to initialize Whisper encoder: result=%d\n", result);
        goto cleanup;
    }
    encoderInitialized = true;
    if ((result = initializeWhisperModel(decoderPath, &appContext.decoderContext)) != 0) {
        std::printf("Failed to initialize Whisper decoder: result=%d\n", result);
        goto cleanup;
    }
    decoderInitialized = true;

    {
        const int updateFrames = kUpdateLengthSeconds * kSampleRate;
        const int windowFrames = chunkLength * kSampleRate;
        int iteration = 0;
        for (int endFrame = std::min(updateFrames, inputAudio.numFrames);
             endFrame > 0;
             endFrame = std::min(endFrame + updateFrames, inputAudio.numFrames)) {
            const int firstFrame = std::max(0, endFrame - windowFrames);
            AudioBuffer activeAudio = {
                std::vector<float>(
                    inputAudio.data.begin() + firstFrame,
                    inputAudio.data.begin() + endFrame),
                endFrame - firstFrame,
                1,
                kSampleRate,
            };
            std::vector<float> audioData(kNumMels * windowFrames / kHopLength, 0.0f);
            TranscriptionHypothesis hypothesis;
            preprocessAudio(
                &activeAudio, melFilters.data(), audioData, chunkLength, enableNeon);
            result = runWhisperInference(
                &appContext,
                audioData,
                vocab.data(),
                taskCode,
                enableTimestamps,
                chunkLength,
                hypothesis);
            if (result != 0) {
                std::printf("Whisper inference failed: result=%d\n", result);
                goto cleanup;
            }
            logHypothesis(++iteration, firstFrame, endFrame, hypothesis);
            if (endFrame == inputAudio.numFrames) {
                break;
            }
        }
    }

cleanup:
    if (decoderInitialized) {
        releaseWhisperModel(&appContext.decoderContext);
    }
    if (encoderInitialized) {
        releaseWhisperModel(&appContext.encoderContext);
    }
    return result == 0 ? 0 : -1;
}
