#include <vector>

#include <gtest/gtest.h>

#include "audio_utils.h"

namespace {

TEST(AudioUtilsTest, ConvertChannelsAveragesStereoSamples)
{
    AudioBuffer audio;
    audio.data = {
        0.25f, 0.75f,
        -0.25f, 0.75f,
        -1.0f, -0.5f,
    };
    audio.numFrames = 3;
    audio.numChannels = 2;
    audio.sampleRate = 16000;

    EXPECT_EQ(convertChannels(&audio), 0);

    EXPECT_EQ(audio.data, std::vector<float>({0.5f, 0.25f, -0.75f}));
}

TEST(AudioUtilsTest, ConvertChannelsCancelsOppositeStereoSamples)
{
    AudioBuffer audio;
    audio.data = {
        0.75f, -0.75f,
        -0.25f, 0.25f,
    };
    audio.numFrames = 2;
    audio.numChannels = 2;

    EXPECT_EQ(convertChannels(&audio), 0);

    ASSERT_EQ(audio.data.size(), 2u);
    EXPECT_FLOAT_EQ(audio.data[0], 0.0f);
    EXPECT_FLOAT_EQ(audio.data[1], 0.0f);
}

TEST(AudioUtilsTest, ConvertChannelsUpdatesChannelMetadataAndPreservesFrameMetadata)
{
    AudioBuffer audio;
    audio.data = {0.0f, 1.0f, 1.0f, 0.0f};
    audio.numFrames = 2;
    audio.numChannels = 2;
    audio.sampleRate = 44100;

    EXPECT_EQ(convertChannels(&audio), 0);

    EXPECT_EQ(audio.numChannels, 1);
    EXPECT_EQ(audio.numFrames, 2);
    EXPECT_EQ(audio.sampleRate, 44100);
    EXPECT_EQ(audio.data.size(), static_cast<std::size_t>(audio.numFrames));
}

TEST(AudioUtilsTest, ConvertChannelsHandlesEmptyStereoBuffer)
{
    AudioBuffer audio;
    audio.numFrames = 0;
    audio.numChannels = 2;
    audio.sampleRate = 16000;

    EXPECT_EQ(convertChannels(&audio), 0);

    EXPECT_TRUE(audio.data.empty());
    EXPECT_EQ(audio.numChannels, 1);
    EXPECT_EQ(audio.numFrames, 0);
    EXPECT_EQ(audio.sampleRate, 16000);
}

TEST(AudioUtilsTest, ResampleAudioUpdatesSampleRateMetadata)
{
    AudioBuffer audio;
    audio.data = {0.0f, 1.0f, 0.0f, -1.0f};
    audio.numFrames = 4;
    audio.numChannels = 1;
    audio.sampleRate = 8000;

    EXPECT_EQ(resampleAudio(&audio, audio.sampleRate, 16000), 0);

    EXPECT_EQ(audio.sampleRate, 16000);
    EXPECT_EQ(audio.numFrames, 8);
    EXPECT_EQ(audio.data.size(), static_cast<std::size_t>(audio.numFrames));
}

}  // namespace
