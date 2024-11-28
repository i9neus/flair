#include "LiftingCodec.cuh"

#include "core/nn/mlp/MLP.cuh"
#include "core/nn/DataLoader.cuh"
//#include "core/math/wavelets/cuda/LiftingMLP.cuh"
#include "core/math/wavelets/2d/StaticDWT2.h"
#include "core/utils/ConsoleUtils.h"
#include "core/utils/ThreadUtils.h"
#include <numeric>
#include <fstream>
#include <random>

namespace Flair
{
    __host__ LiftingCodec::LiftingCodec()
	{
	}

    __host__ Image3f LiftingCodec::WaveletTransform(const Image3f& inputImage, const int direction) const
    {
        Image3f outputImage(inputImage.Width(), inputImage.Height());

        // Functor that extracts a channel from the image, encodes it, then emplaces any generated wavelet coefficients
        std::function<void(int)> EncodeChannelFunctor = [&](int chnlIdx)
        {
            Image1f waveletData = inputImage.ExtractChannel(chnlIdx);

            // In-place transform the image using the DCT
            StaticDWT2<float> dwt(waveletData.Width(), 1);
            switch (direction)
            {
            case 1:
                dwt.Forward(waveletData.Vector()); break;
            case -1:
                dwt.Inverse(waveletData.Vector()); break;
            default:
                Assert(false);
            };

            outputImage.EmplaceChannel(waveletData, chnlIdx);
        };

        std::vector<std::thread> workerThreads;
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            workerThreads.emplace_back(EncodeChannelFunctor, chnlIdx);
            Assert(workerThreads.back().joinable());
        }
        for (auto& t : workerThreads) { t.join(); }

        return outputImage;
    } 

    __host__ MLPDataset LiftingCodec::GenerateTrainingSet(Image1f& waveletImage) const
    {
        Image1f varImage;
        varImage.Resize(waveletImage);

        // Only consider the downsampled region of the image
        ImageRegion region(0, 0, waveletImage.Width() / 2 - 1, waveletImage.Height() / 2 - 1);

        const int numThreads = varImage.GetThreadCount();
        std::vector<float> maxVarMap(numThreads, 0.0f);

        // Map and reduce the variance
        Image1f::ParallelMapFunctor calcVarFunctor = [&](const int x, const int y, const int threadIdx, float* pixel)
        {
            float m = 0, m2 = 0;
            for (int v = -1; v <= 1; ++v)
            {
                for (int u = -1; u <= 1; ++u)
                {
                    const float f = waveletImage.Sample(x + u, y + v);
                    m += f;
                    m2 += f * f;
                }
            }
            *pixel = m2 / 9 - sqr(m / 9); // Variance
            //*pixel = std::sqrt(*pixel); // Standard deviation

            maxVarMap[threadIdx] = std::max(*pixel, maxVarMap[threadIdx]);
        };
        varImage.ParallelMap(calcVarFunctor, region);
        const float maxImageVar = std::reduce(maxVarMap.begin(), maxVarMap.end(), 0.0f, [](float a, float b) -> float { return std::max(a, b); });

        // Create empty histograms
        constexpr int kPDFSize = 100;
        std::vector<float> pdf(kPDFSize), cdf(kPDFSize, 0.0f);
        std::vector<std::vector<std::pair<int, int>>> histogram(kPDFSize);

        // Bucket each pixel based on its relative variance
        Image1f::MapFunctor histVarFunctor = [&](const int x, const int y, float* pixel)
        {
            const int bucketIdx = clamp(int(kPDFSize * (*pixel / maxImageVar)), 0, kPDFSize - 1);
            histogram[bucketIdx].emplace_back(x, y);
        };
        varImage.Map(histVarFunctor, region);

        // Construct and normalise a PDF and CDF based on the inverse bucket size
        constexpr float kMaxPdf = 100.f;
        for (int i = 0; i < kPDFSize; ++i)
        {
            pdf[i] = std::min(kMaxPdf, float(region.Area()) / std::max(1ull, histogram[i].size()));
            cdf[i] = ((i == 0) ? 0.f : cdf[i - 1]) + pdf[i];
        }
        for (int i = 0; i < kPDFSize; ++i) { cdf[i] /= cdf.back(); }

        /*std::ofstream file("C:/Unity/SyntheticGS/Assets/HDRI/Histogram.dat");
        if (file.is_open())
        {
            for (int i = 0; i < kPDFSize; ++i)
            {
                file << tfm::format("%.10f ", pdf[i]);
            }
        }
        file.close();*/

        struct ThreadCtx
        {
            MLPDataset::SampleList inputSamples, targetSamples;
            std::mt19937 mt;
            std::uniform_int_distribution<int> rng;
            inline float RandReal() { return float(rng(mt)) / float(std::numeric_limits<int>::max()); }
            inline int RandInt() { return rng(mt); }
        };

        Threaded<ThreadCtx> runner;
        runner.Initialise([](ThreadCtx& ctx, int i, int N)
            {
                ctx.mt = std::mt19937(std::hash<int>{}(i));
                ctx.rng = std::uniform_int_distribution<int>();
            });

        constexpr int kNumSamples = 10000;
        std::atomic<int> numSamples(0);
        Threaded<ThreadCtx>::Functor sampleFunctor = [&](ThreadCtx& ctx, int i, int N)
        {
            const int numThreadSamples = (kNumSamples * (i + 1) / N) - (kNumSamples * (i) / N);
            ctx.inputSamples.resize(numThreadSamples);
            ctx.targetSamples.resize(numThreadSamples);

            for (int sampleIdx = 0; sampleIdx < numThreadSamples; ++sampleIdx)
            {
                std::vector<std::pair<int, int>>* bucket = nullptr;
                do
                {
                    // Draw a bucket from the density PDF
                    auto lower = std::lower_bound(cdf.begin(), cdf.end(), ctx.RandReal());
                    int bucketIdx = std::distance(cdf.begin(), lower);
                    Assert(bucketIdx < histogram.size());
                    bucket = &histogram[bucketIdx];
                } 
                while (bucket->size() == 0);

                // Draw a sample from the bucket
                auto [x, y] = (*bucket)[ctx.RandInt() % bucket->size()];

                // Jitter it slightly to add random variation
                x += ctx.RandInt() % 4 - 2;
                y += ctx.RandInt() % 4 - 2;                    

                // Construct the sample
                auto inputIt = ctx.inputSamples[sampleIdx].begin();
                auto targetIt = ctx.targetSamples[sampleIdx].begin();
                float mean = 0;
                for (int v = -2; v < 2; ++v)
                {
                    for (int u = -2; u < 2; ++u, ++inputIt, ++targetIt)
                    {
                        *inputIt = waveletImage.Sample(x + u, y + v);
                        mean += *inputIt;

                        *targetIt = waveletImage.Sample(waveletImage.Width() / 2 + x + u, y + v);
                    }
                }
                mean /= 16;

                for (auto& f : ctx.inputSamples[sampleIdx])
                {
                    f = (f - mean) / std::max(1.f, mean);
                }
            }
        };
        runner.Run(sampleFunctor);

        // Combine the samples into a single array
        MLPDataset dataset;
        for (auto& ctx : runner.GetContexts())
        {            
            dataset.inputSamples.insert(dataset.inputSamples.end(), ctx.inputSamples.begin(), ctx.inputSamples.end());
            dataset.targetSamples.insert(dataset.targetSamples.end(), ctx.targetSamples.begin(), ctx.targetSamples.end());
            ctx.inputSamples = MLPDataset::SampleList();
            ctx.targetSamples = MLPDataset::SampleList();
        }

        printf_red("Total samples: %i\n", dataset.inputSamples.size());

        // Render the samples
        for (int i = 0; i < dataset.inputSamples.size(); ++i)
        {
            int x = 4 * (i % (region.Width() / 4));
            int y = 4 * (i / (region.Width() / 4));
            auto inputIt = dataset.inputSamples[i].begin();
            auto targetIt = dataset.targetSamples[i].begin();
            for (int v = 0; v < 4; ++v)
            {
                for (int u = 0; u < 4; ++u, ++inputIt, ++targetIt)
                {
                    *waveletImage.At(x + u, y + v) = *inputIt;
                    *waveletImage.At(x + u, 16 + y + v) = *targetIt;
                }
            }
        }

        return dataset;

        //std::swap(waveletImage, varImage);
    }

    __host__ Image3f LiftingCodec::Encode(const Image3f& inputImage) const
    { 
        Image3f gammaImage = inputImage;

        gammaImage.Saturate();
        //gammaImage.ApplyGamma(1 / 2.2f);
            
        Image3f waveletImage = WaveletTransform(gammaImage, 1);       

        /*for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            Image1f chnlData = waveletImage.ExtractChannel(chnlIdx);

            GenerateTrainingSet(chnlData);

            waveletImage.EmplaceChannel(chnlData, chnlIdx);
        }*/ 

        Image1f chnlData = waveletImage.ExtractChannel(0);

        MLPDataset dataset = GenerateTrainingSet(chnlData);

        NN::MLP mlp;
        mlp.Train(dataset);

        waveletImage.Erase();
        waveletImage.ParallelMap([&](const int x, const int y, const int, float* pixel)
            {
                const float& c = chnlData.At(x, y)[0];
                pixel[(c < 0) ? 0 : 1] = std::abs(c);           
            });        

        return waveletImage;
    }

    __host__ Image3f LiftingCodec::Decode(const Image3f& inputImage) const
    {
        Image3f waveletImage = WaveletTransform(inputImage, -1);

        //waveletImage.ApplyGamma(2.2f);

        return waveletImage;
    }
}
