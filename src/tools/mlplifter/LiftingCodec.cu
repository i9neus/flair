#include "LiftingCodec.cuh"

#include "core/nn/mlp/MLP.cuh"
//#include "core/math/wavelets/cuda/LiftingMLP.cuh"
#include "core/math/wavelets/2d/StaticDWT2.h"
#include "core/utils/ConsoleUtils.h"
#include "core/utils/ThreadUtils.h"
#include "core/image/ImageOps.h"
#include <numeric>
#include <fstream>
#include <random>

namespace Flair
{
    LiftingCodec::LiftingCodec()
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

    __host__ void LiftingCodec::PrepareEncoder(const Image1f& waveletImage)
    {
        ImageRect region(0, 0, waveletImage.Width() / 2, waveletImage.Height() / 2);
        
        // Create a mipmap
        m_mipMap[0] = Crop(waveletImage, region);
        for (int i = 1; i < 4; ++i)
        {
            m_mipMap[i] = Downsample(m_mipMap[i - 1], 2);
        }      
    }

    __host__ std::pair<LiftingCodec::Sample, float> LiftingCodec::GenerateInputSample(const int x, const int y) const
    {
        Sample sample;
        
        /*for (int i = 0; i < 4; ++i)
        {
            float mean = 0;
            int p = i % 2, q = i / 2;
            for (int v = 0; v < 2; ++v)
            {
                for (int u = 0; u < 2; ++u)
                {
                    const float f = m_mipMap[i].Sample(x / (1 << i) + u, y / (1 << i) + v);
                    sample[(q * 2 + v) * 4 + p*2 + u] = f;
                    mean += f;
                }
            }
            
            mean = std::max(1e-3f, mean / 4);
            for (int v = 0; v < 2; ++v)
            {
                for (int u = 0; u < 2; ++u)
                {
                    sample[(q * 2 + v) * 4 + p * 2 + u] -= mean;
                }
            }
        }*/

        auto it = sample.begin();
        float mean = 0;
        float absMax = 0;
        for (int v = 0; v < 4; ++v)
        {
            for (int u = 0; u < 4; ++u, ++it)
            {
                const float f = m_mipMap[0].Sample(x + u, y + v);
                *it = f;
                mean += f;
                absMax = std::max(absMax, std::abs(f));
            }
        }
        mean = std::max(1e-3f, mean / 16);
        absMax = std::max(1e-3f, absMax) * 0.5;
        for (auto& f : sample) { f = (f - mean) / mean; }

        return { sample, mean };
    }

    __host__ void LiftingCodec::GenerateTrainingSet(Image1f& waveletImage, const int basisU, const int basisV, const int seed, LiftingCodec::SampleList& inputSamples, std::vector<float>& inputMeans, LiftingCodec::SampleList& targetSamples) const
    {
        inputSamples.clear();
        inputMeans.clear();
        targetSamples.clear();
        
        Image1f varImage;
        varImage.Resize(waveletImage);

        // Only consider the downsampled region of the image
        ImageRect region(0, 0, waveletImage.Width() / 2 - 4, waveletImage.Height() / 2 - 4);

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
            *pixel = std::pow(*pixel, 0.6f); // Mix
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
            if (x < region.x1 - 4 && y < region.y1 - 4)
            {
                const int bucketIdx = clamp(int(kPDFSize * (*pixel / maxImageVar)), 0, kPDFSize - 1);
                histogram[bucketIdx].emplace_back(x, y);
            }
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
            SampleList inputSamples;
            SampleList targetSamples;
            std::vector<float> inputMeans;
            std::mt19937 mt;
            std::uniform_int_distribution<int> rng;
            inline float RandReal() { return float(rng(mt)) / float(std::numeric_limits<int>::max()); }
            inline int RandInt() { return rng(mt); }
        };

        Threaded<ThreadCtx> runner;
        runner.Initialise([&](ThreadCtx& ctx, int i, int N)
            {
                ctx.mt = std::mt19937(std::hash<int>{}(i + seed));
                ctx.rng = std::uniform_int_distribution<int>();
            });

        constexpr int kNumSamples = 10000;
        std::atomic<int> numSamples(0);
        Threaded<ThreadCtx>::Functor sampleFunctor = [&](ThreadCtx& ctx, int i, int N)
        {
            const int numThreadSamples = (kNumSamples * (i + 1) / N) - (kNumSamples * (i) / N);
            ctx.inputSamples.resize(numThreadSamples);
            ctx.inputMeans.resize(numThreadSamples);
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
                //x += ctx.RandInt() % 2;
                //y += ctx.RandInt() % 2; 

                // Construct the input sample
                auto [ sample, mean ] = GenerateInputSample(x, y);
                ctx.inputSamples[sampleIdx] = sample;
                ctx.inputMeans[sampleIdx] = mean;

                // Construct the output sample
                auto targetIt = ctx.targetSamples[sampleIdx].begin();
                x += basisU;
                y += basisV;

                for (int v = 0; v < 4; ++v)
                {
                    for (int u = 0; u < 4; ++u, ++targetIt)
                    {
                        *targetIt = waveletImage.Sample(x + u, y + v);
                        *targetIt /= mean;
                        //*targetIt = std::pow(std::abs(*targetIt), 1.f) * sign(*targetIt) ;
                    }
                }                
            }
        };
        runner.Run(sampleFunctor);

        // Combine the samples into a single array
        for (auto& ctx : runner.GetContexts())
        {            
            inputSamples.insert(inputSamples.end(), ctx.inputSamples.begin(), ctx.inputSamples.end());
            inputMeans.insert(inputMeans.end(), ctx.inputMeans.begin(), ctx.inputMeans.end());
            targetSamples.insert(targetSamples.end(), ctx.targetSamples.begin(), ctx.targetSamples.end());
            ctx.inputSamples = SampleList();
            ctx.inputMeans = std::vector<float>();
            ctx.targetSamples = SampleList();
        }

        printf_red("Total samples: %i\n", inputSamples.size());

        //std::swap(waveletImage, varImage);
    }

    __host__ Image3f LiftingCodec::Encode(const Image3f& inputImage)
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

        PrepareEncoder(chnlData);

        for (int quadIdx = 1; quadIdx < 1; ++quadIdx)
        {
            const int basisU = (chnlData.Width() / 2) * (quadIdx & 1);
            const int basisV = (chnlData.Height() / 2) * ((quadIdx >> 1) & 1);
            printf_green("Training quadrant [%i, %i]...\n", basisU, basisV);

            ////////////////////////////////////////////////////////////////////////////////////////
            
            printf("Generating training set...\n");
            LiftingCodec::SampleList inputSamples, targetSamples, outputSamples;
            std::vector<float> inputMeans;
            GenerateTrainingSet(chnlData, basisU, basisV, 8783652, inputSamples, inputMeans, targetSamples);

            ////////////////////////////////////////////////////////////////////////////////////////

            printf("Training MLP...\n");
            NN::MLP mlp;
            mlp.Train(inputSamples, targetSamples);

            ////////////////////////////////////////////////////////////////////////////////////////

            ImageRect region(basisU, basisV, basisU + waveletImage.Width() / 2, basisV + waveletImage.Height() / 2);

            /*GenerateTrainingSet(chnlData, basisU, basisV, 235265, inputSamples, inputMeans, targetSamples);

            mlp.Infer( 
                [&](std::vector<Sample>& samples, const int batchIdx) -> bool
                {
                    if (batchIdx != 0) return false;
                    samples = inputSamples;
                    return true;
                },
                [&](const std::vector<Sample>& samples, const int batchIdx) -> void
                {
                    outputSamples = samples;
                }
            );

            // Render the samples
            const int numRows = 1 + int(inputSamples.size() / (waveletImage.Width() / (4 * 2)));
            for (int i = 0; i < inputSamples.size() && i < 10000; ++i)
            {
                int x = 4 * (i % (region.Width() / 4));
                int y = 4 * (i / (region.Width() / 4));
                auto inputIt = inputSamples[i].begin();
                auto targetIt = targetSamples[i].begin();
                auto outputIt = outputSamples[i].begin();
                for (int v = 0; v < 4; ++v)
                {
                    for (int u = 0; u < 4; ++u, ++inputIt, ++targetIt, ++outputIt)
                    {
                        auto Unmap = [&](float value) -> float
                        {
                            //return std::pow(std::abs(value), 1 / 1.f)* sign(value);
                            return value * inputMeans[i];
                        };

                        *chnlData.At(x + u, y + v) = *inputIt;
                        *chnlData.At(x + u, 4 * numRows + y + v) = Unmap(*targetIt);
                        *chnlData.At(x + u, 8 * numRows + y + v) = Unmap(*outputIt);
                    }
                }
            } */

            ////////////////////////////////////////////////////////////////////////////////////////

            const int kSamplesPerBatch = 10000;
            const int kNumPixels = (chnlData.Width() / 2 - 4) * (chnlData.Height() / 2 - 4);
            auto readSamples = [&](std::vector<Sample>& samples, const int batchIdx) -> bool
            {
                if (batchIdx >= kNumPixels)
                {
                    return false;
                }
                else
                {
                    samples.reserve(kSamplesPerBatch);
                    samples.clear();
                    inputMeans.reserve(kSamplesPerBatch);
                    inputMeans.clear();
                    for (int miniBatchIdx = 0, pixelIdx = batchIdx; miniBatchIdx < kSamplesPerBatch && pixelIdx < kNumPixels; ++miniBatchIdx, ++pixelIdx)
                    {
                        const int x = pixelIdx % (chnlData.Width() / 2 - 4), y = pixelIdx / (chnlData.Width() / 2 - 4);
                        auto [sample, mean] = GenerateInputSample(x, y);
                        samples.push_back(sample);
                        inputMeans.push_back(mean);
                    }
                    return true;
                }
            };

            auto writeSamples = [&](const std::vector<Sample>& samples, int batchIdx) -> void
            {
                for (int sampleIdx = 0, pixelIdx = batchIdx; sampleIdx < samples.size(); ++sampleIdx, ++pixelIdx)
                {
                    const int x = pixelIdx % (chnlData.Width() / 2 - 4) + basisU;
                    const int y = pixelIdx / (chnlData.Width() / 2 - 4) + basisV;
                    
                    //*chnlData.At(x, y) = samples[sampleIdx][0] * inputMeans[sampleIdx];
                    auto sampleIt = samples[sampleIdx].begin();
                    for (int v = 0; v < 4; ++v)
                    {
                        for (int u = 0; u < 4; ++u, ++sampleIt)
                        {
                            *chnlData.At(x + u, y + v) += *sampleIt * inputMeans[sampleIdx];
                        }
                    }
                }
            };

            region = ImageRect(basisU, basisV, basisU + waveletImage.Width() / 2, basisV + waveletImage.Height() / 2);
            chnlData.ParallelMap([&](int x, int y, int, float* pixel) { *pixel = 0; }, region);

            printf("Reconstructing wavelet coefficients\n");
            mlp.Infer(readSamples, writeSamples);

            chnlData.ParallelMap([&](int x, int y, int, float* pixel) { *pixel /= 16.; }, region);            
        }

        ImageRect region = ImageRect(0, 0, waveletImage.Width() / 2, waveletImage.Height() / 2);
        //chnlData.ParallelMap([&](int x, int y, int, float* pixel) { *pixel = 0.5; }, region);

        /*waveletImage.Erase();     
        waveletImage.Resize(chnlData);
        waveletImage.ParallelMap([&](const int x, const int y, const int, float* pixel)
            {
                const float& c = chnlData.At(x, y)[0];
                pixel[(c < 0) ? 0 : 1] = std::pow(std::abs(c), 2.0f);
            }); */

        for (int i = 0; i < 3; ++i)
        {
            waveletImage.EmplaceChannel(chnlData, i);
        }

        return waveletImage;
    }

    __host__ Image3f LiftingCodec::Decode(const Image3f& inputImage)
    {
        Image3f waveletImage = WaveletTransform(inputImage, -1);

        //waveletImage.ApplyGamma(2.2f);

        return waveletImage;
    }
}
