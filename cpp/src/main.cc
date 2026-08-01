// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "audio_utils.h"
#include "easy_timer.h"
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
    if (argc != 5) {
        std::printf("%s <encoder_path> <decoder_path> <task> <audio_path>\n", argv[0]);
        return -1;
    }

    const char* encoderPath = argv[1];
    const char* decoderPath = argv[2];
    const char* task = argv[3];
    const char* audioPath = argv[4];
    const char* vocabPath = nullptr;
    int taskCode = 0;

    if (std::strcmp(task, "en") == 0) {
        vocabPath = kEnglishVocabPath;
        taskCode = kEnglishTaskCode;
    } else if (std::strcmp(task, "zh") == 0) {
        vocabPath = kChineseVocabPath;
        taskCode = kChineseTaskCode;
    } else {
        std::printf(
            "\n\033[1;33mCurrently only English or Chinese recognition tasks are supported. "
            "Please specify <task> as en or zh.\033[0m\n");
        return -1;
    }

    int result = 0;
    EasyTimer timer;
    RknnWhisperContext appContext;
    std::vector<float> audioData(kNumMels * kMaxAudioLength / kHopLength, 0.0f);
    std::vector<float> melFilters(kNumMels * kMelFilterSize);
    std::vector<VocabEntry> vocab(kVocabSize);
    std::vector<std::string> recognizedText;
    AudioBuffer audio = {};

    timer.tik();
    result = readAudio(audioPath, &audio);
    if (result != 0) {
        std::printf("Failed to read audio: result=%d, path=%s\n", result, audioPath);
        goto cleanup;
    }

    if (audio.numChannels == 2) {
        result = convertChannels(&audio);
        if (result != 0) {
            std::printf("Failed to convert audio channels: result=%d\n", result);
            goto cleanup;
        }
    }

    if (audio.sampleRate != kSampleRate) {
        result = resampleAudio(&audio, audio.sampleRate, kSampleRate);
        if (result != 0) {
            std::printf("Failed to resample audio: result=%d\n", result);
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
        std::printf(
            "Failed to read Mel filters: result=%d, path=%s\n",
            result,
            kMelFiltersPath);
        goto cleanup;
    }

    result = readVocab(vocabPath, vocab.data());
    if (result != 0) {
        std::printf("Failed to read vocabulary: result=%d, path=%s\n", result, vocabPath);
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Read Mel filters and vocabulary");

    timer.tik();
    result = initializeWhisperModel(encoderPath, &appContext.encoderContext);
    if (result != 0) {
        std::printf(
            "Failed to initialize Whisper encoder: result=%d, path=%s\n",
            result,
            encoderPath);
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Initialize Whisper encoder");

    timer.tik();
    result = initializeWhisperModel(decoderPath, &appContext.decoderContext);
    if (result != 0) {
        std::printf(
            "Failed to initialize Whisper decoder: result=%d, path=%s\n",
            result,
            decoderPath);
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Initialize Whisper decoder");

    timer.tik();
    preprocessAudio(&audio, melFilters.data(), audioData);
    result = runWhisperInference(
        &appContext,
        audioData,
        vocab.data(),
        taskCode,
        recognizedText);
    if (result != 0) {
        std::printf("Whisper inference failed: result=%d\n", result);
        goto cleanup;
    }
    timer.tok();
    timer.printTime("Run Whisper inference");

    std::cout << "\nWhisper output: ";
    for (const auto& text : recognizedText) {
        std::cout << text;
    }
    std::cout << '\n';

    {
        const float inferenceTime = timer.elapsedMs() / 1000.0f;
        const float audioLength = std::min(
            audio.numFrames / static_cast<float>(kSampleRate),
            static_cast<float>(kChunkLength));
        const float realTimeFactor = inferenceTime / audioLength;
        std::printf(
            "\nReal Time Factor (RTF): %.3f / %.3f = %.3f\n",
            inferenceTime,
            audioLength,
            realTimeFactor);
    }

cleanup:
    result = releaseWhisperModel(&appContext.encoderContext);
    if (result != 0) {
        std::printf("Failed to release Whisper encoder: result=%d\n", result);
    }
    result = releaseWhisperModel(&appContext.decoderContext);
    if (result != 0) {
        std::printf("Failed to release Whisper decoder: result=%d\n", result);
    }

    return 0;
}
