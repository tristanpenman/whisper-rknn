#include "audio_utils.h"

#include <cmath>
#include <utility>
#include <vector>

#include <sndfile.h>

#include "logger.h"

int readAudio(const char* path, AudioBuffer* audio)
{
    SF_INFO fileInfo = {0};
    SNDFILE* inputFile = sf_open(path, SFM_READ, &fileInfo);
    if (inputFile == NULL) {
        LOG(ERROR) << "Failed to open audio file '" << path << "': " << sf_strerror(NULL);
        return -1;
    }

    if (fileInfo.channels > 2) {
        LOG(ERROR) << "Unsupported channel count in audio file '" << path << "': "
                   << fileInfo.channels << " (maximum is 2)";
        sf_close(inputFile);
        return -1;
    }

    audio->numFrames = fileInfo.frames;
    audio->numChannels = fileInfo.channels;
    audio->sampleRate = fileInfo.samplerate;
    audio->data.resize(audio->numFrames * audio->numChannels);

    const sf_count_t numReadFrames = sf_readf_float(inputFile, audio->data.data(), audio->numFrames);
    if (numReadFrames != audio->numFrames) {
        LOG(ERROR) << "Failed to read all audio frames: expected " << audio->numFrames
                   << ", got " << numReadFrames;
        audio->data.clear();
        sf_close(inputFile);
        return -1;
    }

    sf_close(inputFile);
    return 0;
}

int saveAudio(
    const char* path,
    const float* data,
    int numFrames,
    int sampleRate,
    int numChannels)
{
    SF_INFO fileInfo = {0};
    fileInfo.frames = numFrames;
    fileInfo.samplerate = sampleRate;
    fileInfo.channels = numChannels;
    fileInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* outputFile = sf_open(path, SFM_WRITE, &fileInfo);
    if (outputFile == NULL) {
        LOG(ERROR) << "Failed to open audio file '" << path << "' for writing: "
                   << sf_strerror(NULL);
        return -1;
    }

    const sf_count_t numWrittenFrames = sf_writef_float(outputFile, data, numFrames);
    if (numWrittenFrames != numFrames) {
        LOG(ERROR) << "Failed to write all audio frames: expected " << numFrames
                   << ", wrote " << numWrittenFrames;
        sf_close(outputFile);
        return -1;
    }

    sf_close(outputFile);
    return 0;
}

int resampleAudio(AudioBuffer* audio, int originalSampleRate, int desiredSampleRate)
{
    const int originalLength = audio->numFrames;
    const int outputLength = std::round(originalLength * (double)desiredSampleRate / (double)originalSampleRate);
    LOG(INFO) << "Resampling audio: " << originalSampleRate << " Hz -> "
              << desiredSampleRate << " Hz";

    std::vector<float> resampledData(outputLength);

    for (int i = 0; i < outputLength; ++i) {
        const double sourceIndex = i * (double)originalSampleRate / (double)desiredSampleRate;
        const int leftIndex = static_cast<int>(std::floor(sourceIndex));
        const int rightIndex = leftIndex + 1 < originalLength ? leftIndex + 1 : leftIndex;
        const double fraction = sourceIndex - leftIndex;
        resampledData[i] = (1.0f - fraction) * audio->data[leftIndex] + fraction * audio->data[rightIndex];
    }

    audio->numFrames = outputLength;
    audio->sampleRate = desiredSampleRate;
    audio->data = std::move(resampledData);
    return 0;
}

int convertChannels(AudioBuffer* audio)
{
    LOG(INFO) << "Converting audio channels: " << audio->numChannels << " -> 1";

    std::vector<float> convertedData(audio->numFrames);

    for (int i = 0; i < audio->numFrames; ++i) {
        const float left = audio->data[i * 2];
        const float right = audio->data[i * 2 + 1];
        convertedData[i] = (left + right) / 2.0f;
    }

    audio->numChannels = 1;
    audio->data = std::move(convertedData);
    return 0;
}
