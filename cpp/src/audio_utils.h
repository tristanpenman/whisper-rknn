#pragma once

#include <vector>

typedef struct AudioBuffer
{
    std::vector<float> data;
    int numFrames = 0;
    int numChannels = 0;
    int sampleRate = 0;
} AudioBuffer;

/** Reads an audio file into a buffer. */
int readAudio(const char* path, AudioBuffer* audio);

/** Saves audio data to a WAV file. */
int saveAudio(
    const char* path,
    const float* data,
    int numFrames,
    int sampleRate,
    int numChannels);

/** Resamples audio data to the desired sample rate. */
int resampleAudio(AudioBuffer* audio, int originalSampleRate, int desiredSampleRate);

/** Converts two-channel audio to mono by averaging both channels. */
int convertChannels(AudioBuffer* audio);
