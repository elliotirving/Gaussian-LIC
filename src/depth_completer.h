/*
 * Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferVersion.h"

// Gaussian-LIC was written against TensorRT 8.6 (x86 desktop). JetPack 6.2 on
// the Jetson Orin ships TensorRT 10.x, which removed the implicit-binding API
// (getNbBindings / getBindingDimensions / executeV2). The implementation below
// is guarded on NV_TENSORRT_MAJOR so the same source compiles on both:
//   - TRT < 10 : legacy binding-index path (executeV2)
//   - TRT >= 10: named-tensor path (setTensorAddress + enqueueV3)
class DepthCompleter
{
public:
    DepthCompleter(const std::string& enginePath,
                   int inputWidth, int inputHeight);
    ~DepthCompleter();

    cv::Mat complete(const cv::Mat& rgbImage, const cv::Mat& depthImage);

    DepthCompleter(const DepthCompleter&) = delete;
    DepthCompleter& operator=(const DepthCompleter&) = delete;

private:
    struct InferDeleter
    {
        template <typename T> void operator()(T* obj) const;
    };

    class Logger : public nvinfer1::ILogger
    {
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
    };

    void initEngine(const std::string& enginePath);
    std::vector<char> readFile(const std::string& filename);
    void allocateBuffers();
    size_t volume(const nvinfer1::Dims& dims);
    void prepareInputs(const cv::Mat& rgbImage, const cv::Mat& depthImage);
    cv::Mat processOutput();

    Logger mLogger;
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> mContext;

    // Index-aligned host/device buffers. For TRT < 10 the indices match engine
    // binding indices (so executeV2 can take the raw vector). For TRT >= 10 the
    // order is arbitrary; the per-tensor names are tracked separately.
    std::vector<void*> mDeviceBuffers;
    std::vector<std::vector<float>> mHostBuffers;
    std::vector<std::string> mTensorNames;   // populated on TRT >= 10

    int mRgbIdx{-1};
    int mDepthIdx{-1};
    int mMaskIdx{-1};
    int mOutputIdx{-1};

    cudaStream_t mStream{nullptr};            // used by enqueueV3 on TRT >= 10
    int mInputWidth, mInputHeight;
};
