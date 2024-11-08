#pragma once

#include "../Codec.h"
#include "../image/Operators.h"
#include <array>

namespace Flair
{
    // Compute the mean square error between two images
    template<int Channels>
    static float ComputeMSE(const Image<float, Channels>& approxImage, const Image<float, Channels>& referenceImage)
    {
        AssertMsg(approxImage.Width() == referenceImage.Width() && approxImage.Height() == referenceImage.Height(), "Images are not the same size.");

        double mse = 0;
        const float* approxData = approxImage.Data();
        const float* refData = referenceImage.Data();
        for (int y = 0, idx = 0; y < referenceImage.Height(); ++y)
        {
            for (int x = 0; x < referenceImage.Width(); ++x)
            {
                for (int c = 0; c < Channels; ++c, ++idx)
                {
                    if (std::isfinite(approxData[idx]) && std::isfinite(refData[idx]))
                    {
                        mse += sqr(approxData[idx] - refData[idx]);
                    }
                }
            }
        }

        return mse / (referenceImage.Area() * Channels);
    }

    // Compute the peak signal-to-noise ratio between two images
    template<int Channels>
    static float ComputePSNR(const Image<float, Channels>& approxImage, const Image<float, Channels>& referenceImage)
    {
        AssertMsg(approxImage.Width() == referenceImage.Width() && approxImage.Height() == referenceImage.Height(), "Images are not the same size.");

        const float* approxData = approxImage.Data();
        const float* refData = referenceImage.Data();
        std::array<double, Channels> mse = {};
        std::array<float, Channels> peak = {};

        for (int y = 0, idx = 0; y < referenceImage.Height(); ++y)
        {
            for (int x = 0; x < referenceImage.Width(); ++x)
            {
                for (int c = 0; c < Channels; ++c, ++idx)
                {
                    if (std::isfinite(approxData[idx]) && std::isfinite(refData[idx]))
                    {
                        mse[c] += sqr(approxData[idx] - refData[idx]);
                        peak[c] = std::max(peak[c], refData[idx]);
                    }
                }
            }
        }

        double meanPSNR = 0.;
        for (int c = 0; c < Channels; ++c)
        {
            meanPSNR += 20 * std::log10(peak[c]) - 10 * std::log10(mse[c] / referenceImage.Area());
        }
        return meanPSNR;
    }

    // Compute the structural similarity index between two images
    static float ComputeSSIM(const Image1f& approxImage, const Image1f& referenceImage, const int patchSize, const int patchStride)
    {
        AssertMsg(approxImage.Width() == referenceImage.Width() && approxImage.Height() == referenceImage.Height(), "Images are not the same size.");
        AssertMsg(patchSize >= 3, "Patch size must be >= 3");
        AssertMsg(patchStride >= 1, "Patch stride must be >= 1");

        float ssim = 0;
        const float kEpsilon1 = sqr(0.01 * 1e-3f), kEpsilon2 = sqr(0.03 * 1e-3f);
        const int width = referenceImage.Width(), height = referenceImage.Height();
        int numPatches = 0;
        for (int j = 0, imgIdx = 0; j < referenceImage.Height() / patchStride; ++j)
        {
            for (int i = 0; i < referenceImage.Width() / patchStride; ++i)
            {
                float x = 0;
                float x2 = 0;
                float y = 0;
                float y2 = 0;
                float xy = 0;
                int N = 0;
                for (int v = -patchSize / 2; v <= patchSize / 2; ++v)
                {
                    for (int u = -patchSize / 2; u <= patchSize / 2; ++u)
                    {
                        if (i + u >= 0 && i + u < width && j + v >= 0 && j + v < height)
                        {
                            const int p = (j + v) * width + (i + u);
                            const auto& fx = approxImage[p];
                            const auto& fy = referenceImage[p];
                            x += fx;
                            x2 += fx * fx;
                            y += fy;
                            y2 += fy * fy;
                            xy += fx * fy;
                            ++N;
                        }
                    }
                }

                x /= N;
                x2 /= N;
                y /= N;
                y2 /= N;
                xy /= N;
                const float varX = x2 - x * x;
                const float varY = y2 - y * y;
                const float covXY = xy - x * y;

                ssim += (2 * x * y + kEpsilon1) * (2 * covXY + kEpsilon2) /
                    ((x * x + y * y + kEpsilon1) * (varX + varY + kEpsilon2));
                ++numPatches;
            }        
        }

        return ssim / numPatches;
    }

    // Compute the multiscale SSIM between two images
    static float ComputeMultiscaleSSIM(const Image1f& approxImage, const Image1f& referenceImage, const int patchSize, const int patchStride)
    {
        AssertMsg(approxImage.Width() == referenceImage.Width() && approxImage.Height() == referenceImage.Height(), "Images are not the same size.");

        float ssim = ComputeSSIM(approxImage, referenceImage, patchSize, patchStride);

        Image1f approxImageDown(approxImage), referenceImageDown(referenceImage);
        int minDim = std::min(referenceImage.Width(), referenceImage.Height()) / 2;

        float sumW = 0;
        for (int i = 0; minDim > 2 * patchSize; ++i, minDim >>= 1)
        {
            approxImageDown = Downsample(approxImageDown, 2);
            referenceImageDown = Downsample(referenceImageDown, 2);

            float w = 1.;// std::pow(2.0f, float(i));
            ssim += ComputeSSIM(approxImage, referenceImage, patchSize, patchStride) * w;
            sumW += w;
        }

        return ssim / sumW;
    }

    struct CodecStats
    {
        size_t      originalSize;
        size_t      compressedSize;
        float       compressionEfficiency;
        float       bitsPerPixel;
        float       mse;
        float       psnr;
        float       ssim;
        float       score;
    };

    // Compute statistics for a compression/decompression cycle 
    template<int Channels>
    static CodecStats GenerateCodecStats(const Image<float, Channels>& approxImage, const Image<float, Channels>& referenceImage, const CompressedImageData& compressedData)
    {
        CodecStats stats;
        stats.originalSize = approxImage.Area() * Channels * sizeof(float);
        stats.compressedSize = compressedData.SizeOf();
        stats.compressionEfficiency = 1 - stats.compressedSize / float(stats.originalSize);
        stats.bitsPerPixel = double(8 * stats.compressedSize) / referenceImage.Area();
        stats.mse = ComputeMSE(approxImage, referenceImage);
        stats.psnr = ComputePSNR(approxImage, referenceImage);
        stats.ssim = ComputeSSIM(approxImage.ExtractLuminance(), referenceImage.ExtractLuminance(), 16, 4);
        stats.score = stats.ssim + 1.f / (1 + stats.bitsPerPixel);
        return stats;
    }

    template<typename... Pack>
    static void PrintKeyValue(const std::string& keyStr, const int keyColWidth, const std::string& valueStr, Pack... pack)
    {
        std::printf("%s", keyStr.c_str());
        for (int i = 0; i < std::max(1, keyColWidth - int(keyStr.size())); ++i) { std::printf(" "); }
        std::printf(valueStr.c_str(), pack...);
    }

    // Print a nicely formatted list of statistics
    static void PrintStats(const CodecStats& stats)
    {
        constexpr int kKeyColWidth = 30;
        PrintKeyValue("  - Compressed size:", kKeyColWidth, "%i bytes\n", stats.compressedSize);
        PrintKeyValue("  - Compression efficiency:", kKeyColWidth, "%.2f%%\n", 100.f * stats.compressionEfficiency);
        PrintKeyValue("  - Bits per pixel:", kKeyColWidth, "%.5f\n", stats.bitsPerPixel);
        PrintKeyValue("  - RMSE:", kKeyColWidth, "%.5f\n", std::sqrt(stats.mse));
        PrintKeyValue("  - PSNR:", kKeyColWidth, "%.5fdB\n", stats.psnr / 3.);
        PrintKeyValue("  - SSIM:", kKeyColWidth, "%.5f\n", stats.ssim);
        PrintKeyValue("  - Score:", kKeyColWidth, "%.5f\n", stats.score);
    }
}