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

#include "whisper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "rknn_utils.h"
#include "string_utils.h"

namespace {

constexpr int kStartOfTranscriptToken = 50258;
constexpr int kEndOfTextToken = 50257;
constexpr int kChineseTaskToken = 50260;
constexpr int kTranscribeToken = 50359;
constexpr int kNoTimestampsToken = 50363;
constexpr int kTimestampBeginToken = 50364;
constexpr int kMaximumDecodeIterations = 1000;

int runEncoder(
    RknnAppContext* appContext,
    const std::vector<float>& audioData,
    int encoderInputSize,
    int encoderOutputSize,
    float* encoderOutput)
{
    rknn_input inputs[1] = {};
    rknn_output outputs[1] = {};

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    std::vector<float> inputData(audioData.begin(), audioData.begin() + kNumMels * encoderInputSize);
    inputs[0].size = inputData.size() * sizeof(float);
    inputs[0].buf = inputData.data();

    int result = rknn_inputs_set(appContext->rknnContext, 1, inputs);
    if (result < 0) {
        std::printf("rknn_inputs_set failed: %s\n", rknn_utils::rknnErrorMessage(result));
        goto cleanup;
    }

    result = rknn_run(appContext->rknnContext, nullptr);
    if (result < 0) {
        std::printf("rknn_run failed: %s\n", rknn_utils::rknnErrorMessage(result));
        goto cleanup;
    }

    outputs[0].want_float = 1;
    result = rknn_outputs_get(appContext->rknnContext, 1, outputs, nullptr);
    if (result < 0) {
        std::printf("rknn_outputs_get failed: %s\n", rknn_utils::rknnErrorMessage(result));
        goto cleanup;
    }

    std::memcpy(encoderOutput, outputs[0].buf, encoderOutputSize * sizeof(float));

cleanup:
    rknn_outputs_release(appContext->rknnContext, 1, outputs);
    return result;
}

int runDecoder(
    RknnAppContext* appContext,
    const float* encoderOutput,
    const VocabEntry* vocab,
    int taskCode,
    bool enableTimestamps,
    int decoderInputSize,
    int maxTokens,
    TranscriptionHypothesis& transcriptionHypothesis)
{
    rknn_input inputs[2] = {};
    rknn_output outputs[1] = {};

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT64;
    std::vector<std::int64_t> tokenInput(maxTokens);
    inputs[0].size = tokenInput.size() * sizeof(std::int64_t);
    inputs[0].buf = tokenInput.data();

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    std::vector<float> encoderInput(encoderOutput, encoderOutput + decoderInputSize);
    inputs[1].size = encoderInput.size() * sizeof(float);
    inputs[1].buf = encoderInput.data();

    const std::array<std::int64_t, 4> initialPrompt = {
        kStartOfTranscriptToken, taskCode, kTranscribeToken, kNoTimestampsToken};
    const int initialPromptLength = enableTimestamps ? 3 : 4;
    int nextToken = kStartOfTranscriptToken;
    int tokenCount = initialPromptLength;
    int iterationCount = 0;

    std::copy_n(initialPrompt.begin(), initialPromptLength, tokenInput.begin());

    int result = 0;
    while (nextToken != kEndOfTextToken && tokenCount < maxTokens && iterationCount < kMaximumDecodeIterations) {
        ++iterationCount;

        result = rknn_inputs_set(appContext->rknnContext, 2, inputs);
        if (result < 0) {
            std::printf("rknn_inputs_set failed: %s\n", rknn_utils::rknnErrorMessage(result));
            goto cleanup;
        }

        result = rknn_run(appContext->rknnContext, nullptr);
        if (result < 0) {
            std::printf("rknn_run failed: %s\n", rknn_utils::rknnErrorMessage(result));
            goto cleanup;
        }

        outputs[0].want_float = 1;
        result = rknn_outputs_get(appContext->rknnContext, 1, outputs, nullptr);
        if (result < 0) {
            std::printf("rknn_outputs_get failed: %s\n", rknn_utils::rknnErrorMessage(result));
            goto cleanup;
        }

        nextToken = argmax(static_cast<const float*>(outputs[0].buf), tokenCount - 1);

        transcriptionHypothesis.tokenIds.push_back(nextToken);
        transcriptionHypothesis.text += vocab[nextToken].token;

        if (nextToken != kEndOfTextToken) {
            tokenInput[tokenCount] = nextToken;
            ++tokenCount;
        }

        rknn_outputs_release(appContext->rknnContext, 1, outputs);
        outputs[0].buf = nullptr;
    }

    replaceSubstring(transcriptionHypothesis.text, "\u0120", " ");
    replaceSubstring(transcriptionHypothesis.text, "<|endoftext|>", "");
    replaceSubstring(transcriptionHypothesis.text, "\n", "");

    if (!transcriptionHypothesis.text.empty()) {
        if (taskCode == kChineseTaskToken) {
            transcriptionHypothesis.text = decodeBase64(transcriptionHypothesis.text);
        }
    }

cleanup:
    return result;
}

}  // namespace

int initializeWhisperModel(const char* modelPath, RknnAppContext* appContext)
{
    rknn_context context = 0;
    int result = rknn_init(&context, const_cast<char*>(modelPath), 0, 0, nullptr);
    if (result < 0) {
        std::printf("rknn_init failed: %s\n", rknn_utils::rknnErrorMessage(result));
        return -1;
    }

    rknn_input_output_num ioCount = {};
    result = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &ioCount, sizeof(ioCount));
    if (result != RKNN_SUCC) {
        std::printf("rknn_query failed: %s\n", rknn_utils::rknnErrorMessage(result));
        rknn_destroy(context);
        return -1;
    }
    rknn_utils::logRknnVersion(context);
    std::printf("model input count: %d, output count: %d\n", ioCount.n_input, ioCount.n_output);

    std::vector<rknn_tensor_attr> inputAttributes(ioCount.n_input);
    std::printf("input tensors:\n");
    for (std::uint32_t i = 0; i < ioCount.n_input; ++i) {
        inputAttributes[i].index = i;
        result = rknn_query(
            context,
            RKNN_QUERY_INPUT_ATTR,
            &inputAttributes[i],
            sizeof(rknn_tensor_attr));
        if (result != RKNN_SUCC) {
            std::printf("rknn_query failed: %s\n", rknn_utils::rknnErrorMessage(result));
            rknn_destroy(context);
            return -1;
        }
        std::printf("  %s\n", rknn_utils::tensorAttrToString(inputAttributes[i]).c_str());
    }

    std::vector<rknn_tensor_attr> outputAttributes(ioCount.n_output);
    std::printf("output tensors:\n");
    for (std::uint32_t i = 0; i < ioCount.n_output; ++i) {
        outputAttributes[i].index = i;
        result = rknn_query(
            context,
            RKNN_QUERY_OUTPUT_ATTR,
            &outputAttributes[i],
            sizeof(rknn_tensor_attr));
        if (result != RKNN_SUCC) {
            std::printf("rknn_query failed: %s\n", rknn_utils::rknnErrorMessage(result));
            rknn_destroy(context);
            return -1;
        }
        std::printf("  %s\n", rknn_utils::tensorAttrToString(outputAttributes[i]).c_str());
    }

    appContext->rknnContext = context;
    appContext->ioCount = ioCount;
    appContext->inputAttributes = std::move(inputAttributes);
    appContext->outputAttributes = std::move(outputAttributes);
    return 0;
}

int releaseWhisperModel(RknnAppContext* appContext)
{
    appContext->inputAttributes.clear();
    appContext->outputAttributes.clear();

    if (appContext->rknnContext != 0) {
        rknn_destroy(appContext->rknnContext);
        appContext->rknnContext = 0;
    }
    return 0;
}

int runWhisperInference(
    RknnWhisperContext* appContext,
    const std::vector<float>& audioData,
    const VocabEntry* vocab,
    int taskCode,
    bool enableTimestamps,
    int chunkLength,
    int maxTokens,
    TranscriptionHypothesis& transcriptionHypothesis)
{
    const int encoderInputSize = chunkLength * 100;
    const int encoderOutputSize = chunkLength * 50 * 512;
    std::vector<float> encoderOutput(encoderOutputSize);
    TranscriptionHypothesis decodedHypothesis;

    transcriptionHypothesis.text.clear();
    transcriptionHypothesis.tokenIds.clear();

    int result = runEncoder(
        &appContext->encoderContext,
        audioData,
        encoderInputSize,
        encoderOutputSize,
        encoderOutput.data());
    if (result != 0) {
        std::printf("Encoder inference failed: %d\n", result);
        return result;
    }

    result = runDecoder(
        &appContext->decoderContext,
        encoderOutput.data(),
        vocab,
        taskCode,
        enableTimestamps,
        encoderOutputSize,
        maxTokens,
        decodedHypothesis);
    if (result != 0) {
        std::printf("Decoder inference failed: %d\n", result);
        return result;
    }

    transcriptionHypothesis = std::move(decodedHypothesis);
    return 0;
}
