#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AudioBuffer
{
    float* data;
    int numFrames;
    int numChannels;
    int sampleRate;
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

#ifdef __cplusplus
}  // extern "C"
#endif
