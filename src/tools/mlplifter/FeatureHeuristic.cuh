#pragma once

#include "core/image/Image.h"
#include <array>

namespace Flair
{
    template<typename Derived>
    class FeatureHeuristic
    { 
    public:
        FeatureHeuristic() = delete;

        static void Classify(const Image1f& inputImage, const ImageRect& region, Image1f& heuristicImage, float& maxVal, const int dilateRadius)
        {
            heuristicImage.ResizeFrom(inputImage);

            const int numThreads = heuristicImage.GetThreadCount();
            std::vector<float> maxVarMap(numThreads, 0.0f);

            // Map and reduce the variance
            Image1f::ParallelMapFunctor calcVarFunctor = [&](const int x, const int y, const int threadIdx, float* pixel)
            {
                *pixel = Derived::EvaluatePixel(inputImage, x, y);

                maxVarMap[threadIdx] = std::max(*pixel, maxVarMap[threadIdx]);
            };
            heuristicImage.ParallelMap(calcVarFunctor, region);

            // Dilate the heuristic map 
            if (dilateRadius > 0)
            {
                Image1f::ParallelMapFunctor dilateFunctor = [&](const int x, const int y, const int threadIdx, float* pixel)
                {
                    float dilated = 0;
                    float mean = 0;
                    int numPixels = 0;
                    for (int v = -dilateRadius; v <= dilateRadius; ++v)
                    {
                        for (int u = -dilateRadius; u <= dilateRadius; ++u)
                        {
                            if (heuristicImage.Contains(x + u, y + v))
                            {
                                const auto f = heuristicImage.At(x + u, y + v)[0];
                                dilated = std::max(dilated, f);
                                mean += f;
                                numPixels++;
                            }
                        }
                        *pixel = std::max(heuristicImage.At(x, y)[0], mean / numPixels);
                        //*pixel = dilated;
                    }
                };

                Image1f swapImage(inputImage.Width(), inputImage.Height());
                swapImage.ParallelMap(dilateFunctor, region);
                heuristicImage = std::move(swapImage);
            }

            maxVal = std::reduce(maxVarMap.begin(), maxVarMap.end(), 0.0f, [](float a, float b) -> float { return std::max(a, b); });
        }
    };
    
    // Measures the variance of the image in a 3x3 window centered around the pixel at x,y
    template<int KernelRadius>
    class VarianceHeuristic : public FeatureHeuristic<VarianceHeuristic<KernelRadius>>
    {
    private:
        enum : int 
        { 
            kKernelSize = KernelRadius * 2 + 1,
            kKernelArea = kKernelSize * kKernelSize
        };

    public:
        VarianceHeuristic() = delete;

        static float EvaluatePixel(const Image1f& inputImage, const int x, const int y)
        {
            float m = 0, m2 = 0;
            for (int v = -KernelRadius; v <= KernelRadius; ++v)
            {
                for (int u = -KernelRadius; u <= KernelRadius; ++u)
                {
                    const float f = inputImage.Sample(x + u, y + v);
                    m += f;
                    m2 += f * f;
                }
            }
            float var = m2 / kKernelArea - sqr(m / kKernelArea); // Variance
            var = std::pow(var, 0.6f); // Mix
            //var = std::sqrt(var); // Standard deviation

            return var;
        }
    };

    // 2D discrete cosine transform with quadratic time complexity 
    template<int KernelRadius>
    class DCTFeatureHeuristic : public FeatureHeuristic<DCTFeatureHeuristic<KernelRadius>>
    {
    private:
        enum : int
        {
            kKernelSize = KernelRadius * 2 + 1,
            kKernelArea = kKernelSize * kKernelSize
        };
        using ValueTable = std::array<float, kKernelArea>;

    private:
        static ValueTable GetValueTable(const Image1f& inputImage, const int x, const int y)
        {
            ValueTable values;
            float meanVal = 0;
            for (int v = -KernelRadius + 1, i = 0; v <= KernelRadius; ++v)
            {
                for (int u = -KernelRadius + 1; u <= KernelRadius; ++u, ++i)
                {
                    // The DCT requires a 4x4 grid however we only want 3x3. Clamp the pixel coordinates to the so that we don't accidentally 
                    // sample pixels belonging to features that fall outside of the input window of the NN model
                    values[i] = inputImage.Sample(x + std::min(KernelRadius - 1, u), y + std::min(KernelRadius - 1, v));
                    //values[i] = inputImage.Sample(x + u, y + v);
                    meanVal += values[i];
                }
            }

            // Normalise the values based on the mean
            meanVal /= kKernelArea;
            for (auto& v : values) { v /= std::max(1e-3f, meanVal); }

            return values;
        }

        static ValueTable ForwardDCT2D(const ValueTable& values)
        {
            // Dumb n^2 complexity DCT. Requires kKernelArea^2 iterations per pixel for a window size of . 
            // TODO: Optimise using Cooley-Tukey to get complexity down to linearithmic complexity.
            ValueTable coeffs;
            for (int l = 0, i = 0; l < kKernelSize; ++l)
            {
                for (int m = 0; m < kKernelSize; ++m, ++i)
                {
                    float sigma = 0., mean = 0.;
                    float norm = (1. + float(l)) * (1. + float(m));
                    for (int v = -KernelRadius, j = 0; v < KernelRadius; ++v)
                    {
                        for (int u = -KernelRadius; u < KernelRadius; ++u, ++j)
                        {
                            float coeff = std::cos(kTwoPi * float(l) * 0.5 * (float(u + KernelRadius) + 0.5) / float(kKernelSize)) *
                                          std::cos(kTwoPi * float(m) * 0.5 * (float(v + KernelRadius) + 0.5) / float(kKernelSize)) * norm;

                            sigma += values[j] * coeff;
                        }
                    }
                    if (i != 0) sigma /= coeffs[0];
                    coeffs[i] = sigma / (float(kKernelArea));
                }
            }
            return coeffs;
        }

    public:
        DCTFeatureHeuristic() = delete;

        static float EvaluatePixel(const Image1f& inputImage, const int x, const int y)
        {
            // Do forward 2D DCT on the table of input coefficients
            ValueTable coeffs = ForwardDCT2D(GetValueTable(inputImage, x, y));

            // Measure the variance
            float m = 0., m2 = 0.;
            for (int i = 1; i < kKernelArea; ++i)
            {
                m += abs(coeffs[i]);
                m2 += sqr((coeffs[i]));
            }
            m /= float(kKernelArea - 1);
            m2 /= float(kKernelArea - 1);
            const float var = m2 - sqr(m);

            // The final heuristic is the sum of the standard deviation and the mean, normalised by the DC coefficient
            return (sqrt(var) * m) / std::max(1e-2f, coeffs[0]);
        }
    };

}