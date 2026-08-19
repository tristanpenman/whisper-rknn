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
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "logger.h"
#include "rknn_utils.h"
#include "string_utils.h"

namespace {

constexpr int kStartOfTranscriptToken = 50258;
constexpr int kEndOfTextToken = 50257;
constexpr int kChineseTaskToken = 50260;
constexpr int kTranscribeToken = 50359;
constexpr int kNoTimestampsToken = 50363;
constexpr int kTimestampBeginToken = 50364;
constexpr int kMaxInitialTimestampIndex = 50;
constexpr int kMaxDecodeIterations = 1000;

int timestampArgmax(const float* decoderOutput, int tokenIndex, const std::vector<int>& generatedTokens)
{
    const int outputOffset = tokenIndex * kVocabSize;
    const float* logits = decoderOutput + outputOffset;
    const bool isFirstToken = generatedTokens.empty();
    const bool lastWasTimestamp =
        !isFirstToken && generatedTokens.back() >= kTimestampBeginToken;
    const bool penultimateWasTimestamp =
        generatedTokens.size() < 2 || generatedTokens[generatedTokens.size() - 2] >= kTimestampBeginToken;

    const float negativeInfinity = -std::numeric_limits<float>::infinity();
    std::vector<float> filteredLogits(logits, logits + kVocabSize);
    filteredLogits[kNoTimestampsToken] = negativeInfinity;

    if (isFirstToken) {
        std::fill(filteredLogits.begin(), filteredLogits.begin() + kTimestampBeginToken, negativeInfinity);
        const int firstDisallowedTimestamp = kTimestampBeginToken + kMaxInitialTimestampIndex + 1;
        std::fill(filteredLogits.begin() + firstDisallowedTimestamp, filteredLogits.end(), negativeInfinity);
        const auto maximum = std::max_element(filteredLogits.begin(), filteredLogits.end());
        return static_cast<int>(maximum - filteredLogits.begin());
    }

    if (lastWasTimestamp) {
        if (penultimateWasTimestamp) {
            std::fill(filteredLogits.begin() + kTimestampBeginToken, filteredLogits.end(), negativeInfinity);
        } else {
            std::fill(filteredLogits.begin(), filteredLogits.begin() + kEndOfTextToken, negativeInfinity);
        }
    }

    auto lastTimestamp = std::find_if(generatedTokens.rbegin(), generatedTokens.rend(), [](int token) {
        return token >= kTimestampBeginToken;
    });
    if (lastTimestamp != generatedTokens.rend()) {
        int minimumTimestamp = *lastTimestamp;
        if (!lastWasTimestamp || penultimateWasTimestamp) {
            ++minimumTimestamp;
        }
        minimumTimestamp = std::min(minimumTimestamp, kVocabSize);
        std::fill(
            filteredLogits.begin() + kTimestampBeginToken,
            filteredLogits.begin() + minimumTimestamp,
            negativeInfinity);
    }

    const auto timestampBegin = filteredLogits.begin() + kTimestampBeginToken;
    const float maxTimestampLogit = *std::max_element(timestampBegin, filteredLogits.end());
    if (std::isfinite(maxTimestampLogit)) {
        float timestampProbabilitySum = 0.0f;
        for (auto iterator = timestampBegin; iterator != filteredLogits.end(); ++iterator) {
            timestampProbabilitySum += std::exp(*iterator - maxTimestampLogit);
        }
        const float timestampLogProbability = maxTimestampLogit + std::log(timestampProbabilitySum);
        const float maxTextLogit = *std::max_element(filteredLogits.begin(), timestampBegin);
        if (timestampLogProbability > maxTextLogit) {
            std::fill(filteredLogits.begin(), timestampBegin, negativeInfinity);
        }
    }

    const auto maximum = std::max_element(filteredLogits.begin(), filteredLogits.end());
    return static_cast<int>(maximum - filteredLogits.begin());
}

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
        LOG(ERROR) << "rknn_inputs_set failed: " << rknn_utils::rknnErrorMessage(result);
        goto cleanup;
    }

    result = rknn_run(appContext->rknnContext, nullptr);
    if (result < 0) {
        LOG(ERROR) << "rknn_run failed: " << rknn_utils::rknnErrorMessage(result);
        goto cleanup;
    }

    outputs[0].want_float = 1;
    result = rknn_outputs_get(appContext->rknnContext, 1, outputs, nullptr);
    if (result < 0) {
        LOG(ERROR) << "rknn_outputs_get failed: " << rknn_utils::rknnErrorMessage(result);
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
    TranscriptionHypothesis& transcriptionHypothesis)
{
    const int initialPromptLength = enableTimestamps ? 3 : 4;

    // The decoder graph is exported with a fixed token input length, so the
    // model itself determines how many tokens can be decoded.
    if (appContext->inputAttributes.empty()) {
        LOG(ERROR) << "Decoder model does not report any input tensors";
        return -1;
    }
    const int tokenInputLength = static_cast<int>(appContext->inputAttributes[0].n_elems);
    if (tokenInputLength <= initialPromptLength) {
        LOG(ERROR) << "Decoder token input length is " << tokenInputLength << ", but at least "
                   << initialPromptLength + 1 << " is required";
        return -1;
    }

    rknn_input inputs[2] = {};
    rknn_output outputs[1] = {};

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT64;
    std::vector<std::int64_t> tokenInput(tokenInputLength);
    inputs[0].size = tokenInput.size() * sizeof(std::int64_t);
    inputs[0].buf = tokenInput.data();

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    std::vector<float> encoderInput(encoderOutput, encoderOutput + decoderInputSize);
    inputs[1].size = encoderInput.size() * sizeof(float);
    inputs[1].buf = encoderInput.data();

    const std::array<std::int64_t, 4> initialPrompt = {
        kStartOfTranscriptToken, taskCode, kTranscribeToken, kNoTimestampsToken
    };

    int nextToken = kStartOfTranscriptToken;
    int tokenCount = initialPromptLength;
    int iterationCount = 0;

    std::copy_n(initialPrompt.begin(), initialPromptLength, tokenInput.begin());

    int result = 0;
    while (nextToken != kEndOfTextToken && tokenCount < tokenInputLength && iterationCount < kMaxDecodeIterations) {
        ++iterationCount;

        result = rknn_inputs_set(appContext->rknnContext, 2, inputs);
        if (result < 0) {
            LOG(ERROR) << "rknn_inputs_set failed: " << rknn_utils::rknnErrorMessage(result);
            goto cleanup;
        }

        result = rknn_run(appContext->rknnContext, nullptr);
        if (result < 0) {
            LOG(ERROR) << "rknn_run failed: " << rknn_utils::rknnErrorMessage(result);
            goto cleanup;
        }

        outputs[0].want_float = 1;
        result = rknn_outputs_get(appContext->rknnContext, 1, outputs, nullptr);
        if (result < 0) {
            LOG(ERROR) << "rknn_outputs_get failed: " << rknn_utils::rknnErrorMessage(result);
            goto cleanup;
        }

        const float* decoderOutput = static_cast<const float*>(outputs[0].buf);
        nextToken = enableTimestamps
            ? timestampArgmax(decoderOutput, tokenCount - 1, transcriptionHypothesis.tokenIds)
            : argmax(decoderOutput, tokenCount - 1);

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
        LOG(ERROR) << "rknn_init failed: " << rknn_utils::rknnErrorMessage(result);
        return -1;
    }

    rknn_input_output_num ioCount = {};
    result = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &ioCount, sizeof(ioCount));
    if (result != RKNN_SUCC) {
        LOG(ERROR) << "rknn_query failed: " << rknn_utils::rknnErrorMessage(result);
        rknn_destroy(context);
        return -1;
    }
    rknn_utils::logRknnVersion(context);
    LOG(INFO) << "Model input count: " << ioCount.n_input << ", output count: " << ioCount.n_output;

    std::vector<rknn_tensor_attr> inputAttributes(ioCount.n_input);
    LOG(INFO) << "Input tensors:";
    for (std::uint32_t i = 0; i < ioCount.n_input; ++i) {
        inputAttributes[i].index = i;
        result = rknn_query(
            context,
            RKNN_QUERY_INPUT_ATTR,
            &inputAttributes[i],
            sizeof(rknn_tensor_attr));
        if (result != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed: " << rknn_utils::rknnErrorMessage(result);
            rknn_destroy(context);
            return -1;
        }
        LOG(INFO) << "  " << rknn_utils::tensorAttrToString(inputAttributes[i]);
    }

    std::vector<rknn_tensor_attr> outputAttributes(ioCount.n_output);
    LOG(INFO) << "Output tensors:";
    for (std::uint32_t i = 0; i < ioCount.n_output; ++i) {
        outputAttributes[i].index = i;
        result = rknn_query(
            context,
            RKNN_QUERY_OUTPUT_ATTR,
            &outputAttributes[i],
            sizeof(rknn_tensor_attr));
        if (result != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed: " << rknn_utils::rknnErrorMessage(result);
            rknn_destroy(context);
            return -1;
        }
        LOG(INFO) << "  " << rknn_utils::tensorAttrToString(outputAttributes[i]);
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

int getWhisperChunkLength(const RknnAppContext& encoderContext, int* chunkLength)
{
    if (chunkLength == nullptr || encoderContext.inputAttributes.empty()) {
        LOG(ERROR) << "Encoder model does not report an input tensor";
        return -1;
    }

    constexpr int kMelFramesPerSecond = kSampleRate / kHopLength;
    constexpr int kElementsPerSecond = kNumMels * kMelFramesPerSecond;
    const std::uint32_t inputElements = encoderContext.inputAttributes[0].n_elems;
    if (inputElements == 0 || inputElements % kElementsPerSecond != 0) {
        LOG(ERROR) << "Encoder input has " << inputElements
                   << " elements; expected 80 Mel bins and a whole number of seconds";
        return -1;
    }

    *chunkLength = static_cast<int>(inputElements / kElementsPerSecond);
    return 0;
}

int runWhisperInference(
    RknnWhisperContext* appContext,
    const std::vector<float>& audioData,
    const VocabEntry* vocab,
    int taskCode,
    bool enableTimestamps,
    TranscriptionHypothesis& transcriptionHypothesis)
{
    if (appContext->encoderContext.inputAttributes.empty()
        || appContext->encoderContext.outputAttributes.empty()) {
        LOG(ERROR) << "Encoder model does not report its tensor dimensions";
        return -1;
    }
    const int encoderInputSize =
        static_cast<int>(appContext->encoderContext.inputAttributes[0].n_elems / kNumMels);
    const int encoderOutputSize =
        static_cast<int>(appContext->encoderContext.outputAttributes[0].n_elems);
    if (audioData.size() < static_cast<std::size_t>(kNumMels * encoderInputSize)) {
        LOG(ERROR) << "Mel spectrogram is smaller than the encoder input tensor";
        return -1;
    }
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
        LOG(ERROR) << "Encoder inference failed: " << result;
        return result;
    }

    result = runDecoder(
        &appContext->decoderContext,
        encoderOutput.data(),
        vocab,
        taskCode,
        enableTimestamps,
        encoderOutputSize,
        decodedHypothesis);
    if (result != 0) {
        LOG(ERROR) << "Decoder inference failed: " << result;
        return result;
    }

    transcriptionHypothesis = std::move(decodedHypothesis);
    return 0;
}
