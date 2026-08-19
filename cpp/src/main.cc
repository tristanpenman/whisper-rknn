// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Copyright (c) 2026 by Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
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
    std::vector<float> audioData;
    std::vector<float> melFilters(kNumMels * kMelFilterSize);
    std::vector<VocabEntry> vocab(kVocabSize);
    TranscriptionHypothesis transcriptionHypothesis;
    AudioBuffer audio = {};

    timer.tik();
    result = readAudio(audioPath, &audio);
    if (result != 0) {
        LOG(ERROR) << "Failed to read audio: result=" << result << ", path=" << audioPath;
        goto cleanup;
    }

    if (audio.numChannels == 2) {
        result = convertChannels(&audio);
        if (result != 0) {
            LOG(ERROR) << "Failed to convert audio channels: result=" << result;
            goto cleanup;
        }
    }

    if (audio.sampleRate != kSampleRate) {
        result = resampleAudio(&audio, audio.sampleRate, kSampleRate);
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
    audioData.resize(kNumMels * chunkLength * kSampleRate / kHopLength, 0.0f);
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

    timer.tik();
    preprocessAudio(&audio, melFilters.data(), audioData, chunkLength, enableNeon);
    result = runWhisperInference(
        &appContext,
        audioData,
        vocab.data(),
        taskCode,
        enableTimestamps,
        transcriptionHypothesis);
    if (result != 0) {
        LOG(ERROR) << "Whisper inference failed: result=" << result;
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Run Whisper inference");

    std::cout << "\nWhisper output token IDs: ";
    for (const auto& tokenId : transcriptionHypothesis.tokenIds) {
        std::cout << tokenId << ' ';
    }
    std::cout << '\n';

    std::cout << "\nWhisper output text: ";
    std::cout << transcriptionHypothesis.text << '\n';

    {
        const float inferenceTime = timer.elapsedMs() / 1000.0f;
        const float audioLength = std::min(
            audio.numFrames / static_cast<float>(kSampleRate),
            static_cast<float>(chunkLength));
        const float realTimeFactor = inferenceTime / audioLength;
        LOG(INFO) << std::fixed << std::setprecision(3)
                  << "Real Time Factor (RTF): " << inferenceTime << " / " << audioLength
                  << " = " << realTimeFactor;
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
