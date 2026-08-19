// Copyright (c) 2026 by Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

#include "audio_utils.h"
#include "easy_timer.h"
#include "logger.h"
#include "process.h"
#include "whisper.h"

namespace {

constexpr char kEnglishVocabPath[] = "./model/vocab_en.txt";
constexpr char kChineseVocabPath[] = "./model/vocab_zh.txt";
constexpr int kEnglishTaskCode = 50259;
constexpr int kChineseTaskCode = 50260;
constexpr int kUpdateLengthSeconds = 5;
constexpr int kContextLengthSeconds = 1;
constexpr int kEndOfTextToken = 50257;
constexpr int kTimestampBeginToken = 50364;
constexpr int kTimestampFrames = kSampleRate / 50;

int firstCompletedSegmentFrames(
    const std::vector<int>& tokenIds,
    int minimumEndFrame)
{
    bool hasText = false;
    for (const int tokenId : tokenIds) {
        if (tokenId >= kTimestampBeginToken) {
            if (hasText) {
                const int endFrame = (tokenId - kTimestampBeginToken) * kTimestampFrames;
                if (endFrame > minimumEndFrame) {
                    return endFrame;
                }
                hasText = false;
            }
        } else if (tokenId != kEndOfTextToken) {
            hasText = true;
        }
    }

    return 0;
}

void logHypothesis(
    int iteration,
    int firstFrame,
    int endFrame,
    const TranscriptionHypothesis& hypothesis)
{
    LOG(INFO) << std::fixed << std::setprecision(1) << "Streaming iteration " << iteration
              << " (audio " << firstFrame / static_cast<double>(kSampleRate) << "-"
              << endFrame / static_cast<double>(kSampleRate) << " seconds)";
    std::ostringstream tokenIds;
    for (const int tokenId : hypothesis.tokenIds) {
        tokenIds << tokenId << ' ';
    }
    LOG(INFO) << "Whisper output token IDs: " << tokenIds.str();
    LOG(INFO) << "Whisper output text: " << hypothesis.text;
}

}  // namespace

int main(int argc, char** argv)
{
    Logger::configure();

    bool enableNeon = true;
    bool enableTimestamps = false;
    int argumentOffset = 0;
    while (argumentOffset + 1 < argc) {
        const char* argument = argv[argumentOffset + 1];
        if (std::strcmp(argument, "--disable-neon") == 0) {
            enableNeon = false;
        } else if (std::strcmp(argument, "--enable-timestamps") == 0) {
            enableTimestamps = true;
        } else {
            break;
        }
        ++argumentOffset;
    }

    if (argc != 5 + argumentOffset) {
        LOG(ERROR) << "Usage: " << argv[0]
                   << " [--disable-neon] [--enable-timestamps]"
                      " <encoder_path> <decoder_path> <task> <audio_path>";
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
        LOG(ERROR) << "Currently only English or Chinese recognition tasks are supported. "
                      "Please specify <task> as en or zh.";
        return -1;
    }

    int result = 0;
    int chunkLength = 0;
    EasyTimer timer;
    RknnWhisperContext appContext;
    std::vector<float> melFilters(kNumMels * kMelFilterSize);
    std::vector<VocabEntry> vocab(kVocabSize);
    AudioBuffer inputAudio = {};

    timer.tik();
    result = readAudio(audioPath, &inputAudio);
    if (result != 0) {
        LOG(ERROR) << "Failed to read audio: result=" << result << ", path=" << audioPath;
        goto cleanup;
    }

    if (inputAudio.numChannels == 2) {
        result = convertChannels(&inputAudio);
        if (result != 0) {
            LOG(ERROR) << "Failed to convert audio channels: result=" << result;
            goto cleanup;
        }
    }

    if (inputAudio.sampleRate != kSampleRate) {
        result = resampleAudio(&inputAudio, inputAudio.sampleRate, kSampleRate);
        if (result != 0) {
            LOG(ERROR) << "Failed to resample audio: result=" << result;
            goto cleanup;
        }
    }
    timer.tok();
    timer.printTime("Read and normalize audio");

    timer.tik();
    result = readMelFilters(
        kMelFiltersPath,
        melFilters.data(),
        static_cast<int>(melFilters.size()));
    if (result != 0) {
        LOG(ERROR) << "Failed to read Mel filters: result=" << result
                   << ", path=" << kMelFiltersPath;
        goto cleanup;
    }

    result = readVocab(vocabPath, vocab.data());
    if (result != 0) {
        LOG(ERROR) << "Failed to read vocabulary: result=" << result << ", path=" << vocabPath;
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Read Mel filters and vocabulary");

    timer.tik();
    result = initializeWhisperModel(encoderPath, &appContext.encoderContext);
    if (result != 0) {
        LOG(ERROR) << "Failed to initialize Whisper encoder: result=" << result
                   << ", path=" << encoderPath;
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Initialize Whisper encoder");

    result = getWhisperChunkLength(appContext.encoderContext, &chunkLength);
    if (result != 0) {
        goto cleanup;
    }
    if (chunkLength < kUpdateLengthSeconds) {
        LOG(ERROR) << "Model chunk length must be at least " << kUpdateLengthSeconds << " seconds";
        result = -1;
        goto cleanup;
    }
    LOG(INFO) << "Model chunk length: " << chunkLength << " seconds";

    timer.tik();
    result = initializeWhisperModel(decoderPath, &appContext.decoderContext);
    if (result != 0) {
        LOG(ERROR) << "Failed to initialize Whisper decoder: result=" << result
                   << ", path=" << decoderPath;
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Initialize Whisper decoder");

    {
        const int updateFrames = kUpdateLengthSeconds * kSampleRate;
        const int windowFrames = chunkLength * kSampleRate;
        const int contextFrames = kContextLengthSeconds * kSampleRate;
        int firstFrame = 0;
        int endFrame = std::min(updateFrames, inputAudio.numFrames);
        int committedEndFrame = 0;
        int iteration = 0;
        while (endFrame > firstFrame) {
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
            timer.tik();
            preprocessAudio(
                &activeAudio, melFilters.data(), audioData, chunkLength, enableNeon);
            result = runWhisperInference(
                &appContext,
                audioData,
                vocab.data(),
                taskCode,
                enableTimestamps,
                hypothesis);
            if (result != 0) {
                LOG(ERROR) << "Whisper inference failed: result=" << result;
                goto cleanup;
            }
            timer.tok();
            timer.printTime("Run Whisper inference");
            logHypothesis(++iteration, firstFrame, endFrame, hypothesis);

            const int segmentEndFrame = std::min(
                firstCompletedSegmentFrames(
                    hypothesis.tokenIds,
                    committedEndFrame - firstFrame),
                endFrame - firstFrame);
            if (segmentEndFrame > 0) {
                committedEndFrame = firstFrame + segmentEndFrame;
                firstFrame = std::max(0, committedEndFrame - contextFrames);
            }

            if (endFrame < inputAudio.numFrames && endFrame - firstFrame < windowFrames) {
                endFrame = std::min(
                    {endFrame + updateFrames, firstFrame + windowFrames, inputAudio.numFrames});
            } else if (segmentEndFrame == 0) {
                break;
            }
        }
    }

cleanup:
    result = releaseWhisperModel(&appContext.encoderContext);
    if (result != 0) {
        LOG(ERROR) << "Failed to release Whisper encoder: result=" << result;
    }
    result = releaseWhisperModel(&appContext.decoderContext);
    if (result != 0) {
        LOG(ERROR) << "Failed to release Whisper decoder: result=" << result;
    }

    return 0;
}
