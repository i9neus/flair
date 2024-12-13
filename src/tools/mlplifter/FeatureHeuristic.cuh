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

        static void Classify(const Image1f& inputImage, const ImageRect& region, Image1f& varImage, float& maxVal)
        {
            varImage.Resize(inputImage);

            const int numThreads = varImage.GetThreadCount();
            std::vector<float> maxVarMap(numThreads, 0.0f);

            // Map and reduce the variance
            Image1f::ParallelMapFunctor calcVarFunctor = [&](const int x, const int y, const int threadIdx, float* pixel)
            {
                *pixel = Derived::EvaluatePixel(inputImage, x, y);

                maxVarMap[threadIdx] = std::max(*pixel, maxVarMap[threadIdx]);
            };
            varImage.ParallelMap(calcVarFunctor, region);
            maxVal = std::reduce(maxVarMap.begin(), maxVarMap.end(), 0.0f, [](float a, float b) -> float { return std::max(a, b); });
        }
    };
    
    // Measures the variance of the image in a 3x3 window centered around the pixel at x,y
    class VarianceHeuristic : public FeatureHeuristic<VarianceHeuristic>
    {
    public:
        VarianceHeuristic() = delete;

        static float EvaluatePixel(const Image1f& inputImage, const int x, const int y)
        {
            float m = 0, m2 = 0;
            for (int v = -1; v <= 1; ++v)
            {
                for (int u = -1; u <= 1; ++u)
                {
                    const float f = inputImage.Sample(x + u, y + v);
                    m += f;
                    m2 += f * f;
                }
            }
            float var = m2 / 9 - sqr(m / 9); // Variance
            var = std::pow(var, 0.6f); // Mix
            //var = std::sqrt(var); // Standard deviation

            return var;
        }
    };
    
    // 2D discrete cosine transform with quadratic time complexity 
    class DCTFeatureHeuristic : public FeatureHeuristic<DCTFeatureHeuristic>
    {
    private:
        enum : int { kNumCoeffs = 4, kNumCoeffsSqr = kNumCoeffs * kNumCoeffs };
        using ValueTable = std::array<float, kNumCoeffsSqr>;

    private:
        static ValueTable GetValueTable(const Image1f& inputImage, const int x, const int y)
        {
            ValueTable values;
            float meanVal = 0;
            for (int v = -kNumCoeffs / 2 + 1, i = 0; v <= kNumCoeffs / 2; ++v)
            {
                for (int u = -kNumCoeffs / 2 + 1; u <= kNumCoeffs / 2; ++u, ++i)
                {
                    // The DCT requires a 4x4 grid however we only want 3x3. Clamp the pixel coordinates to the so that we don't accidentally 
                    // sample pixels belonging to features that fall outside of the input window of the NN model
                    values[i] = inputImage.Sample(x + std::min(kNumCoeffs / 2 - 1, u), y + std::min(kNumCoeffs / 2 - 1, v));
                    //values[i] = inputImage.Sample(x + u, y + v);
                    meanVal += values[i];
                }
            }
            
            // Normalise the values based on the mean
            meanVal /= kNumCoeffsSqr;
            for (auto& v : values) { v /= std::max(1e-3f, meanVal); }

            return values;
        }

        static ValueTable ForwardDCT2D(const ValueTable& values)
        {
            // Dumb n^2 complexity DCT. Requires 256 iterations per pixel for a window size of 4x4. 
            // TODO: Optimise using Cooley-Tukey to get complexity down to linearithmic complexity.
            ValueTable coeffs;
            for (int l = 0, i = 0; l < kNumCoeffs; ++l)
            {
                for (int m = 0; m < kNumCoeffs; ++m, ++i)
                {
                    float sigma = 0., mean = 0.;
                    float norm = (1. + float(l)) * (1. + float(m));
                    for (int v = -kNumCoeffs / 2, j = 0; v < kNumCoeffs / 2; ++v)
                    {
                        for (int u = -kNumCoeffs / 2; u < kNumCoeffs / 2; ++u, ++j)
                        {
                            float coeff = cos(kTwoPi * float(l) * 0.5 * (float(u + kNumCoeffs / 2) + 0.5) / float(kNumCoeffs)) *
                                cos(kTwoPi * float(m) * 0.5 * (float(v + kNumCoeffs / 2) + 0.5) / float(kNumCoeffs)) * norm;

                            sigma += values[j] * coeff;
                        }
                    }
                    if (i != 0) sigma /= coeffs[0];
                    coeffs[i] = sigma / (float(kNumCoeffsSqr));
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
            for (int i = 1; i < kNumCoeffsSqr; ++i)
            {
                m += abs(coeffs[i]);
                m2 += sqr((coeffs[i]));
            }
            m /= float(kNumCoeffsSqr - 1);
            m2 /= float(kNumCoeffsSqr - 1);
            const float var = m2 - sqr(m);

            // The final heuristic is the sum of the standard deviation and the mean, normalised by the DC coefficient
            return (sqrt(var) * m) / std::max(1e-2f, coeffs[0]);
        }
    };

}