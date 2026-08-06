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

#include "process.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <fftw3.h>

#if ENABLE_NEON
#include <arm_neon.h>
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

void padMelSpectrogram(
    const std::vector<float>& input,
    int inputRows,
    int inputColumns,
    std::vector<float>& output,
    int outputColumns)
{
    for (int row = 0; row < inputRows; ++row) {
        std::copy(
            input.begin() + row * inputColumns,
            input.begin() + (row + 1) * inputColumns,
            output.begin() + row * outputColumns);
    }
}

void makeHannWindow(std::vector<float>& window, int length)
{
    for (int i = 0; i < length; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0 * kPi * i / (length - 1)));
    }
}

void reflectPad(
    const std::vector<float>& audio,
    std::vector<float>& paddedAudio,
    int paddingWidth)
{
    std::copy(audio.begin(), audio.end(), paddedAudio.begin() + paddingWidth);
    std::reverse_copy(audio.begin(), audio.begin() + paddingWidth, paddedAudio.begin());
    std::reverse_copy(audio.end() - paddingWidth, audio.end(), paddedAudio.end() - paddingWidth);
}

#if ENABLE_NEON
void computeStftsNeon(
    const std::vector<float>& audio,
    int audioLength,
    int windowLength,
    int hopLength,
    const std::vector<float>& window,
    fftwf_complex* stftResult,
    int numFrames)
{
    auto* input = static_cast<float*>(fftwf_malloc(sizeof(float) * windowLength));
    auto* output = static_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * (windowLength / 2 + 1)));
    const fftwf_plan plan = fftwf_plan_dft_r2c_1d(windowLength, input, output, FFTW_ESTIMATE);

    for (int frame = 0; frame < numFrames; ++frame) {
        const int start = frame * hopLength;
        for (int i = 0; i < windowLength - 3; i += 4) {
            if (start + i < audioLength) {
                const float32x4_t samples = vld1q_f32(audio.data() + start + i);
                const float32x4_t windowValues = vld1q_f32(window.data() + i);
                vst1q_f32(input + i, vmulq_f32(samples, windowValues));
            } else {
                vst1q_f32(input + i, vdupq_n_f32(0.0f));
            }
        }

        for (int i = windowLength - windowLength % 4; i < windowLength; ++i) {
            if (start + i < audioLength) {
                input[i] = audio[start + i] * window[i];
            } else {
                input[i] = 0.0f;
            }
        }

        fftwf_execute(plan);
        std::memcpy(
            stftResult + frame * (windowLength / 2 + 1),
            output,
            sizeof(fftwf_complex) * (windowLength / 2 + 1));
    }

    fftwf_free(input);
    fftwf_free(output);
    fftwf_destroy_plan(plan);
}
#else
void computeStfts(
    const std::vector<float>& audio,
    int audioLength,
    int windowLength,
    int hopLength,
    const std::vector<float>& window,
    fftwf_complex* stftResult,
    int numFrames)
{
    auto* input = static_cast<float*>(fftwf_malloc(sizeof(float) * windowLength));
    auto* output = static_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * (windowLength / 2 + 1)));
    const fftwf_plan plan = fftwf_plan_dft_r2c_1d(windowLength, input, output, FFTW_ESTIMATE);

    for (int frame = 0; frame < numFrames; ++frame) {
        const int start = frame * hopLength;
        for (int i = 0; i < windowLength; ++i) {
            if (start + i < audioLength) {
                input[i] = audio[start + i] * window[i];
            } else {
                input[i] = 0.0f;
            }
        }

        fftwf_execute(plan);
        std::memcpy(
            stftResult + frame * (windowLength / 2 + 1),
            output,
            sizeof(fftwf_complex) * (windowLength / 2 + 1));
    }

    fftwf_free(input);
    fftwf_free(output);
    fftwf_destroy_plan(plan);
}
#endif

float computeMagnitude(const fftwf_complex& value)
{
    return value[0] * value[0] + value[1] * value[1];
}

void computeMagnitudes(
    const fftwf_complex* stftResult,
    int numMelFilters,
    int numFrames,
    std::vector<float>& magnitudes)
{
    int outputIndex = 0;
    for (int filter = 0; filter < numMelFilters; ++filter) {
        for (int frame = 0; frame < numFrames - 1; ++frame) {
            magnitudes[outputIndex] = computeMagnitude(stftResult[filter * numFrames + frame]);
            ++outputIndex;
        }
    }
}

void clampAndNormalize(std::vector<float>& melSpectrogram, int rows, int columns)
{
    constexpr float kMinimumValue = 1e-10f;
    constexpr float kScalingFactor = 0.25f;
    constexpr float kShiftValue = 4.0f;

    float maxValue = melSpectrogram[0];
    for (int i = 0; i < rows * columns; ++i) {
        const float value = std::max(melSpectrogram[i], kMinimumValue);
        melSpectrogram[i] = std::log10(value);
        if (melSpectrogram[i] > maxValue) {
            maxValue = melSpectrogram[i];
        }
    }

    const float threshold = maxValue - 8.0f;
    for (int i = 0; i < rows * columns; ++i) {
        melSpectrogram[i] =
            (std::max(melSpectrogram[i], threshold) + kShiftValue) * kScalingFactor;
    }
}

void transposeComplex(
    const fftwf_complex* input,
    int inputRows,
    int inputColumns,
    fftwf_complex* output)
{
    for (int row = 0; row < inputRows; ++row) {
        for (int column = 0; column < inputColumns; ++column) {
            const int inputIndex = row * inputColumns + column;
            const int outputIndex = column * inputRows + row;
            output[outputIndex][0] = input[inputIndex][0];
            output[outputIndex][1] = input[inputIndex][1];
        }
    }
}

#if ENABLE_NEON
void multiplyMatricesNeon(
    const float* left,
    const float* right,
    std::vector<float>& output,
    int leftRows,
    int sharedColumns,
    int rightColumns)
{
    for (int row = 0; row < leftRows; ++row) {
        int column = 0;
        for (; column <= rightColumns - 4; column += 4) {
            float32x4_t total = vdupq_n_f32(0.0f);
            for (int shared = 0; shared < sharedColumns; ++shared) {
                const float weight = left[row * sharedColumns + shared];
                if (weight != 0.0f) {
                    const float32x4_t rightValues =
                        vld1q_f32(&right[shared * rightColumns + column]);
                    total = vmlaq_n_f32(total, rightValues, weight);
                }
            }
            vst1q_f32(&output[row * rightColumns + column], total);
        }
        for (; column < rightColumns; ++column) {
            float total = 0.0f;
            for (int shared = 0; shared < sharedColumns; ++shared) {
                total += left[row * sharedColumns + shared]
                    * right[shared * rightColumns + column];
            }
            output[row * rightColumns + column] = total;
        }
    }
}
#endif

void multiplyMatricesScalar(
    const float* left,
    const float* right,
    std::vector<float>& output,
    int leftRows,
    int sharedColumns,
    int rightColumns)
{
    std::fill(output.begin(), output.end(), 0.0f);

    for (int row = 0; row < leftRows; ++row) {
        float* outputRow = output.data() + row * rightColumns;

        for (int shared = 0; shared < sharedColumns; ++shared) {
            const float weight = left[row * sharedColumns + shared];
            if (weight == 0.0f) {
                continue;
            }

            const float* rightRow = right + shared * rightColumns;
            for (int column = 0; column < rightColumns; ++column) {
                outputRow[column] += weight * rightRow[column];
            }
        }
    }
}

void computeLogMelSpectrogram(
    const float* audioData,
    int audioLength,
    int numStftFrames,
    const float* filters,
    std::vector<float>& melSpectrogram,
    bool enableNeon)
{
    std::vector<float> window(kFftSize);
    makeHannWindow(window, kFftSize);

    std::vector<float> audio(audioData, audioData + audioLength);
    std::vector<float> paddedAudio(audioLength + kFftSize);
    reflectPad(audio, paddedAudio, kFftSize / 2);

    auto* stftResult = static_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * kMelFilterSize * numStftFrames));
#if ENABLE_NEON
    computeStftsNeon(
        paddedAudio,
        audioLength + kFftSize,
        kFftSize,
        kHopLength,
        window,
        stftResult,
        numStftFrames);
#else
    computeStfts(
        paddedAudio,
        audioLength + kFftSize,
        kFftSize,
        kHopLength,
        window,
        stftResult,
        numStftFrames);
#endif

    auto* transposedStftResult = static_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * kMelFilterSize * numStftFrames));
    transposeComplex(stftResult, numStftFrames, kMelFilterSize, transposedStftResult);

    std::vector<float> magnitudes(kMelFilterSize * (numStftFrames - 1));
    computeMagnitudes(transposedStftResult, kMelFilterSize, numStftFrames, magnitudes);

    constexpr int kMelRows = kNumMels;
    const int melColumns = numStftFrames - 1;
    multiplyMatrices(
        filters,
        magnitudes.data(),
        melSpectrogram,
        kMelRows,
        kMelFilterSize,
        melColumns,
        enableNeon);

    clampAndNormalize(melSpectrogram, kMelRows, melColumns);
    fftwf_free(stftResult);
    fftwf_free(transposedStftResult);
}

}  // namespace

void multiplyMatrices(
    const float* left,
    const float* right,
    std::vector<float>& output,
    int leftRows,
    int sharedColumns,
    int rightColumns,
    bool enableNeon)
{
#if ENABLE_NEON
    if (enableNeon) {
        multiplyMatricesNeon(
            left, right, output, leftRows, sharedColumns, rightColumns);
        return;
    }
#else
    (void)enableNeon;
#endif
    multiplyMatricesScalar(left, right, output, leftRows, sharedColumns, rightColumns);
}

int readMelFilters(const char* fileName, float* data, int maxLines)
{
    FILE* file = std::fopen(fileName, "r");
    if (file == nullptr) {
        std::perror("Error opening file");
        return -1;
    }

    int lineCount = 0;
    while (lineCount < maxLines && std::fscanf(file, "%f", &data[lineCount]) == 1) {
        ++lineCount;
    }

    std::fclose(file);
    return 0;
}

void preprocessAudio(
    AudioBuffer* audio,
    const float* melFilters,
    std::vector<float>& melSpectrogram,
    int chunkLength,
    bool enableNeon)
{
    const int maxAudioLength = chunkLength * kSampleRate;
    const int audioLength = audio->numFrames;
    std::vector<float> originalAudioData(audio->data.begin(), audio->data.begin() + audioLength);

    if (audioLength >= maxAudioLength) {
        std::vector<float> trimmedAudioData(maxAudioLength);
        std::copy(
            originalAudioData.begin(),
            originalAudioData.begin() + maxAudioLength,
            trimmedAudioData.begin());
        const int numStftFrames = maxAudioLength / kHopLength + 1;
        computeLogMelSpectrogram(
            trimmedAudioData.data(),
            maxAudioLength,
            numStftFrames,
            melFilters,
            melSpectrogram,
            enableNeon);
        return;
    }

    const int numStftFrames = audioLength / kHopLength + 1;
    constexpr int kMelRows = kNumMels;
    const int melColumns = numStftFrames - 1;
    const int paddedMelColumns = maxAudioLength / kHopLength;
    std::vector<float> currentMelSpectrogram(kMelRows * melColumns, 0.0f);
    computeLogMelSpectrogram(
        originalAudioData.data(),
        audioLength,
        numStftFrames,
        melFilters,
        currentMelSpectrogram,
        enableNeon);
    padMelSpectrogram(
        currentMelSpectrogram,
        kMelRows,
        melColumns,
        melSpectrogram,
        paddedMelColumns);
}

int readVocab(const char* fileName, VocabEntry* vocab)
{
    FILE* file = std::fopen(fileName, "r");
    if (file == nullptr) {
        std::perror("Error opening file");
        return -1;
    }

    char line[512];
    int count = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        // The vocabulary format is an integer index followed by a token.
        vocab[count].token = std::strchr(line, ' ') + 1;
        vocab[count].index = std::atoi(line);
        ++count;
    }

    std::fclose(file);
    return 0;
}

int argmax(const float* array, int tokenIndex)
{
    const int startIndex = tokenIndex * kVocabSize;
    int maxIndex = startIndex;
    float maxValue = array[startIndex];
    for (int i = startIndex + 1; i < startIndex + kVocabSize; ++i) {
        if (array[i] > maxValue) {
            maxValue = array[i];
            maxIndex = i;
        }
    }
    return maxIndex - startIndex;
}
