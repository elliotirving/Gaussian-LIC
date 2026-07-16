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

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "cuda_runtime_api.h"
#include "depth_completer.h"

#define TRT_GE_10 (NV_TENSORRT_MAJOR >= 10)

namespace
{
// Classify an engine I/O tensor by the names exported in ckpt/export_onnx_*.py
// (inputs: "rgb", "depth", "mask"; output: "pred").
enum class TensorRole { kRgb, kDepth, kMask, kOutput, kUnknown };

TensorRole classify(const std::string& name, bool isInput)
{
    if (name.find("rgb") != std::string::npos)   return TensorRole::kRgb;
    if (name.find("depth") != std::string::npos) return TensorRole::kDepth;
    if (name.find("mask") != std::string::npos)  return TensorRole::kMask;
    if (!isInput || name.find("pred") != std::string::npos) return TensorRole::kOutput;
    return TensorRole::kUnknown;
}
}  // namespace

void DepthCompleter::Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
    {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

template <typename T>
void DepthCompleter::InferDeleter::operator()(T* obj) const
{
    // TensorRT 10 made the interface objects ordinary deletable objects; the
    // legacy ->destroy() path was removed, so delete works on both versions.
    if (obj) delete obj;
}

DepthCompleter::DepthCompleter(const std::string& enginePath,
                               int inputWidth, int inputHeight)
    : mInputWidth(inputWidth),
      mInputHeight(inputHeight)
{
    initEngine(enginePath);
}

DepthCompleter::~DepthCompleter()
{
    mContext.reset();
    mEngine.reset();
    mRuntime.reset();

    if (mStream) cudaStreamDestroy(mStream);

    for (auto& buf : mDeviceBuffers)
    {
        if (buf) cudaFree(buf);
    }
}

cv::Mat DepthCompleter::complete(const cv::Mat& rgbImage, const cv::Mat& depthImage)
{
    cv::Mat processedRgb, processedDepth;
    // rgbImage.convertTo(processedRgb, CV_32F, 1.0f / 255.0f);
    processedRgb = rgbImage;
    depthImage.convertTo(processedDepth, CV_32F, 1.0f / 200.0f);
    prepareInputs(processedRgb, processedDepth);

#if TRT_GE_10
    if (!mContext->enqueueV3(mStream))
    {
        throw std::runtime_error("Failed to execute inference (enqueueV3)");
    }
    cudaStreamSynchronize(mStream);
#else
    if (!mContext->executeV2(mDeviceBuffers.data()))
    {
        throw std::runtime_error("Failed to execute inference (executeV2)");
    }
#endif

    return processOutput();
}

void DepthCompleter::initEngine(const std::string& enginePath)
{
    auto engineData = readFile(enginePath);
    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!mEngine) throw std::runtime_error("Failed to deserialize engine");
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext) throw std::runtime_error("Failed to create execution context");

    allocateBuffers();
}

std::vector<char> DepthCompleter::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Unable to open file: " + filename);

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read file: " + filename);

    return buffer;
}

void DepthCompleter::allocateBuffers()
{
#if TRT_GE_10
    const int numTensors = mEngine->getNbIOTensors();
#else
    const int numTensors = mEngine->getNbBindings();
#endif
    mDeviceBuffers.assign(numTensors, nullptr);
    mHostBuffers.resize(numTensors);
    mTensorNames.assign(numTensors, std::string());

    for (int i = 0; i < numTensors; i++)
    {
#if TRT_GE_10
        const char* name = mEngine->getIOTensorName(i);
        const bool isInput =
            mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
#else
        const char* name = mEngine->getBindingName(i);
        const bool isInput = mEngine->bindingIsInput(i);
#endif
        mTensorNames[i] = name ? name : "";
        const TensorRole role = classify(mTensorNames[i], isInput);

        // The SPNet engine is built with min == opt == max shapes (see
        // ckpt/build_trt.sh), so every dimension is known here from H/W and the
        // channel count of the role; we size buffers directly rather than trust
        // possibly-dynamic engine dims.
        const int channels = (role == TensorRole::kRgb) ? 3 : 1;
        const size_t elementCount =
            static_cast<size_t>(channels) * mInputHeight * mInputWidth;

        mHostBuffers[i].resize(elementCount);
        if (cudaMalloc(&mDeviceBuffers[i], elementCount * sizeof(float)) != cudaSuccess)
        {
            throw std::runtime_error("CUDA memory allocation failed");
        }

        switch (role)
        {
            case TensorRole::kRgb:    mRgbIdx = i;    break;
            case TensorRole::kDepth:  mDepthIdx = i;  break;
            case TensorRole::kMask:   mMaskIdx = i;   break;
            case TensorRole::kOutput: mOutputIdx = i; break;
            default: break;
        }

#if TRT_GE_10
        if (isInput)
        {
            const nvinfer1::Dims4 shape{1, channels, mInputHeight, mInputWidth};
            if (!mContext->setInputShape(name, shape))
            {
                throw std::runtime_error("Failed to set input shape for tensor: " + mTensorNames[i]);
            }
        }
        if (!mContext->setTensorAddress(name, mDeviceBuffers[i]))
        {
            throw std::runtime_error("Failed to bind device address for tensor: " + mTensorNames[i]);
        }
#endif
    }

    if (mRgbIdx < 0 || mDepthIdx < 0 || mMaskIdx < 0 || mOutputIdx < 0)
    {
        throw std::runtime_error(
            "Engine is missing one of the expected tensors (rgb/depth/mask/pred)");
    }

#if TRT_GE_10
    if (cudaStreamCreate(&mStream) != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA stream");
    }
#endif
}

size_t DepthCompleter::volume(const nvinfer1::Dims& dims)
{
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; i++) v *= dims.d[i];
    return v;
}

void DepthCompleter::prepareInputs(const cv::Mat& rgbImage, const cv::Mat& depthImage)
{
    const size_t plane = static_cast<size_t>(mInputHeight) * mInputWidth;

    // RGB (HWC -> CHW)
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(rgbImage, rgbChannels);
    for (int c = 0; c < 3; c++)
    {
        std::memcpy(mHostBuffers[mRgbIdx].data() + c * plane,
                    rgbChannels[c].data,
                    plane * sizeof(float));
    }

    // Depth
    std::memcpy(mHostBuffers[mDepthIdx].data(), depthImage.data, plane * sizeof(float));

    // Mask
    cv::Mat mask = depthImage > 0;  // CV_8U  0｜255
    mask.convertTo(mask, CV_32F, 1.0 / 255.0);
    std::memcpy(mHostBuffers[mMaskIdx].data(), mask.data, plane * sizeof(float));

    // Copy inputs to device (output buffer is filled by inference).
    const int inputIdx[3] = {mRgbIdx, mDepthIdx, mMaskIdx};
    for (int idx : inputIdx)
    {
        if (cudaMemcpy(mDeviceBuffers[idx], mHostBuffers[idx].data(),
                       mHostBuffers[idx].size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }
}

cv::Mat DepthCompleter::processOutput()
{
    if (cudaMemcpy(mHostBuffers[mOutputIdx].data(), mDeviceBuffers[mOutputIdx],
                   mHostBuffers[mOutputIdx].size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        throw std::runtime_error("CUDA memcpy failed");
    }

    cv::Mat result(mInputHeight, mInputWidth, CV_32F, mHostBuffers[mOutputIdx].data());

    return result * 200.0f;
}
