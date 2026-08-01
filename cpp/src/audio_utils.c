#include "audio_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <sndfile.h>

int readAudio(const char* path, AudioBuffer* audio)
{
    SF_INFO fileInfo = {0};
    SNDFILE* inputFile = sf_open(path, SFM_READ, &fileInfo);
    if (inputFile == NULL) {
        fprintf(stderr, "Error: failed to open file '%s': %s\n", path, sf_strerror(NULL));
        return -1;
    }

    audio->numFrames = fileInfo.frames;
    audio->numChannels = fileInfo.channels;
    audio->sampleRate = fileInfo.samplerate;
    audio->data = (float*)malloc(audio->numFrames * audio->numChannels * sizeof(float));
    if (audio->data == NULL) {
        fprintf(stderr, "Error: failed to allocate memory.\n");
        sf_close(inputFile);
        return -1;
    }

    const sf_count_t numReadFrames = sf_readf_float(inputFile, audio->data, audio->numFrames);
    if (numReadFrames != audio->numFrames) {
        fprintf(
            stderr,
            "Error: failed to read all frames. Expected %ld, got %ld.\n",
            (long)audio->numFrames,
            (long)numReadFrames);
        free(audio->data);
        audio->data = NULL;
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
        fprintf(stderr, "Error: failed to open file '%s' for writing: %s\n", path, sf_strerror(NULL));
        return -1;
    }

    const sf_count_t numWrittenFrames = sf_writef_float(outputFile, data, numFrames);
    if (numWrittenFrames != numFrames) {
        fprintf(
            stderr,
            "Error: failed to write all frames. Expected %ld, wrote %ld.\n",
            (long)numFrames,
            (long)numWrittenFrames);
        sf_close(outputFile);
        return -1;
    }

    sf_close(outputFile);
    return 0;
}

int resampleAudio(AudioBuffer* audio, int originalSampleRate, int desiredSampleRate)
{
    const int originalLength = audio->numFrames;
    const int outputLength = round(originalLength * (double)desiredSampleRate / (double)originalSampleRate);
    printf("resampleAudio: %d Hz -> %d Hz\n", originalSampleRate, desiredSampleRate);

    float* resampledData = (float*)malloc(outputLength * sizeof(float));
    if (resampledData == NULL) {
        return -1;
    }

    for (int i = 0; i < outputLength; ++i) {
        const double sourceIndex = i * (double)originalSampleRate / (double)desiredSampleRate;
        const int leftIndex = (int)floor(sourceIndex);
        const int rightIndex = leftIndex + 1 < originalLength ? leftIndex + 1 : leftIndex;
        const double fraction = sourceIndex - leftIndex;
        resampledData[i] = (1.0f - fraction) * audio->data[leftIndex] + fraction * audio->data[rightIndex];
    }

    audio->numFrames = outputLength;
    free(audio->data);
    audio->data = resampledData;
    return 0;
}

int convertChannels(AudioBuffer* audio)
{
    printf("convertChannels: %d -> 1\n", audio->numChannels);

    float* convertedData = (float*)malloc(audio->numFrames * sizeof(float));
    if (convertedData == NULL) {
        return -1;
    }

    for (int i = 0; i < audio->numFrames; ++i) {
        const float left = audio->data[i * 2];
        const float right = audio->data[i * 2 + 1];
        convertedData[i] = (left + right) / 2.0f;
    }

    audio->numChannels = 1;
    free(audio->data);
    audio->data = convertedData;
    return 0;
}
