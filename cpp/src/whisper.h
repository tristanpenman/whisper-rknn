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

#pragma once

#include <string>
#include <vector>

#include "rknn_api.h"

#include "process.h"

struct RknnAppContext
{
    rknn_context rknnContext = 0;
    rknn_input_output_num ioCount{};
    rknn_tensor_attr* inputAttributes = nullptr;
    rknn_tensor_attr* outputAttributes = nullptr;
};

struct RknnWhisperContext
{
    RknnAppContext encoderContext;
    RknnAppContext decoderContext;
};

int initializeWhisperModel(const char* modelPath, RknnAppContext* appContext);
int releaseWhisperModel(RknnAppContext* appContext);
int runWhisperInference(
    RknnWhisperContext* appContext,
    const std::vector<float>& audioData,
    const VocabEntry* vocab,
    int taskCode,
    std::vector<std::string>& recognizedText);
