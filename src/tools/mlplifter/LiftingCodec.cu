#include "LiftingCodec.cuh"

#include "core/nn/mlp/MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/math/wavelets/2d/StaticDWT2.h"
#include "core/utils/ConsoleUtils.h"
#include "core/utils/ThreadUtils.h"
#include "core/image/ImageOps.h"
#include "FeatureHeuristic.cuh"
#include <numeric>
#include <fstream>
#include <random>

namespace Flair
{
    LiftingCodec::LiftingCodec()
    {
    }

    __host__ void WaveletTransform(Image1f& inputImage, const int direction, const int maxRecurse)
    {
        // In-place transform the image using the DCT   
        StaticDWT2<float> dwt(inputImage.Width(), maxRecurse);
        switch (direction)
        {
        case 1:
            dwt.Forward(inputImage.Vector()); break;
        case -1:
            dwt.Inverse(inputImage.Vector()); break;
        default:
            Assert(false);
        };
    }

    template<int Channels>
    __host__ Image<float, Channels> WaveletTransform(const Image<float, Channels>& inputImage, const int direction, const int maxRecurse)
    {        
        using ImageType = Image<float, Channels>;
        ImageType outputImage(inputImage.Width(), inputImage.Height());

        if (Channels == 1)
        {
            outputImage = inputImage;
            WaveletTransform(outputImage, direction, maxRecurse);
        }
        else
        {
            // Functor that extracts a channel from the image, encodes it, then emplaces any generated wavelet coefficients
            std::function<void(int)> EncodeChannelFunctor = [&](int chnlIdx)
            {
                Image1f waveletData = inputImage.ExtractChannel(chnlIdx);

                WaveletTransform(waveletData, direction, maxRecurse);

                outputImage.EmplaceChannel(waveletData, chnlIdx);
            };

            std::vector<std::thread> workerThreads;
            for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
            {
                workerThreads.emplace_back(EncodeChannelFunctor, chnlIdx);
                Assert(workerThreads.back().joinable());
            }
            for (auto& t : workerThreads) { t.join(); }
        }

        return outputImage;
    } 

    __host__ void LiftingCodec::PrepareEncoder(const Image1f& waveletImage)
    {
        ImageRect region(0, 0, waveletImage.Width() / 2, waveletImage.Height() / 2);
        
        // Create a mipmap
        m_mipMap[0] = waveletImage;
        m_mipMap[1] = Crop(waveletImage, region);

        WaveletTransform(m_mipMap[1], 1, 1);
    }

    __host__ float LiftingCodec::CoefficientAt(float u, float v, const int quadX, const int quadY, const int mipLevel) const
    {
        return m_mipMap[mipLevel].Sample<kImageNearest>(u * 0.5 + quadX * 0.5, v * 0.5 + quadY * 0.5);
    }

    __host__ std::pair<LiftingCodec::InputSample, float> LiftingCodec::GenerateInputSample(const int x, const int y) const
    {
        InputSample sample;        
        float mean = 0; 
        auto it = sample.begin();
        for (int i = 0; i < 4; ++i)
        {
            int quadX = i % 2, quadY = i / 2;
            for (int v = -1; v <= 1; ++v)
            {
                for (int u = -1; u <= 1; ++u, ++it)
                {
                    const float f = CoefficientAt((x + u) / float(m_mipMap[0].Width() / 2), 
                                                  (y + v) / float(m_mipMap[0].Height() / 2), 
                                                  quadX, quadY, (i == 0) ? 0 : 1);                    
                    *it = f; 
                    if(i == 0) mean += f;
                }
            }
        }
        
        mean = std::max(1e-10f, mean / 9);
        for (int i = 0; i < InputSample::kN; ++i) 
        { 
            if (i < 9) { sample[i] -= mean; }
            sample[i] /= mean; 
        }

        return { sample, mean };
    }

    __host__ LiftingCodec::OutputSample LiftingCodec::GenerateTargetSample(const int x, const int y) const
    {
        OutputSample sample;
        auto it = sample.begin();
        for (int i = 1; i < 4; ++i)
        {
            int quadX = i % 2, quadY = i / 2;
            for (int v = -1; v <= 1; ++v)
            {
                for (int u = -1; u <= 1; ++u, ++it)
                {
                    Assert(it != sample.end());
                    const float f = CoefficientAt((x + u) / float(m_mipMap[0].Width() / 2), 
                                                  (y + v) / float(m_mipMap[0].Height() / 2), 
                                                  quadX, quadY, 0);
                    *it = f;
                }
            }
        }
        return sample;
    }

    __host__ void LiftingCodec::GenerateTrainingSet(Image1f& waveletImage, const int seed, LiftingCodec::InputSampleList& inputSamples, std::vector<float>& inputMeans, LiftingCodec::OutputSampleList& targetSamples)
    {
        inputSamples.clear();
        inputMeans.clear();
        targetSamples.clear();
        
        // Only consider the downsampled region of the image
        ImageRect region(0, 0, waveletImage.Width() / 2, waveletImage.Height() / 2);

        float maxFeatureVal;
        using Heuristic = DCTFeatureHeuristic;
        //using Heuristic = VarianceHeuristic;

        Heuristic::Classify(waveletImage, region, m_heuristicImage, maxFeatureVal);

        // Create empty histograms
        constexpr int kPDFSize = 100;
        std::vector<float> pdf(kPDFSize), cdf(kPDFSize, 0.0f);
        std::vector<std::vector<std::pair<int, int>>> histogram(kPDFSize);

        // Bucket each pixel based on its relative variance
        Image1f::MapFunctor histVarFunctor = [&](const int x, const int y, float* pixel)
        {
            if (x < region.x1 && y < region.y1)
            {
                float importance = saturate(std::pow(*pixel / maxFeatureVal, 0.5f) * 2.f);
                const int bucketIdx = clamp(int(kPDFSize * importance), 0, kPDFSize - 1);
                histogram[bucketIdx].emplace_back(x, y);
            }
        };
        m_heuristicImage.Map(histVarFunctor, region);

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
            InputSampleList inputSamples;
            OutputSampleList targetSamples;
            std::vector<float> inputMeans;
            std::mt19937 mt;
            std::uniform_int_distribution<int> rng;
            inline float RandReal() { return float(rng(mt)) / float(std::numeric_limits<int>::max()); }
            inline int RandInt() { return rng(mt); }
        };

        constexpr int kNumSamples = 20000;

        Threaded<ThreadCtx> runner(std::min(16, kNumSamples));
        runner.Initialise([&](ThreadCtx& ctx, int i, int N)
            {
                ctx.mt = std::mt19937(std::hash<int>{}(i + seed));
                ctx.rng = std::uniform_int_distribution<int>();
            });

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
                } while (bucket->size() == 0);

                // Draw a sample from the bucket
                auto [x, y] = (*bucket)[ctx.RandInt() % bucket->size()];

                // Jitter it slightly to add random variation
                //x += ctx.RandInt() % 2;
                //y += ctx.RandInt() % 2; 

                // Construct the input sample
                auto [inputSample, mean] = GenerateInputSample(x, y);
                ctx.inputSamples[sampleIdx] = inputSample;
                ctx.inputMeans[sampleIdx] = mean;
                // ...and the target sample
                ctx.targetSamples[sampleIdx] = GenerateTargetSample(x, y) / mean;
            }
        };
        runner.RunSerial(sampleFunctor);

        // Combine the samples into a single array
        for (auto& ctx : runner.GetContexts())
        {
            inputSamples.insert(inputSamples.end(), ctx.inputSamples.begin(), ctx.inputSamples.end());
            inputMeans.insert(inputMeans.end(), ctx.inputMeans.begin(), ctx.inputMeans.end());
            targetSamples.insert(targetSamples.end(), ctx.targetSamples.begin(), ctx.targetSamples.end());
            ctx.inputSamples = InputSampleList();
            ctx.inputMeans = std::vector<float>();
            ctx.targetSamples = OutputSampleList();
        }

        printf_red("Total samples: %i\n", inputSamples.size());

        //std::swap(waveletImage, m_heuristicImage);
    }

    template<typename SampleType>
    __host__ int RenderSampleBlock(Image1f& image, const SampleType& sample, int x, int y)
    {
        auto inputIt = sample.begin();
        int j = 0;
        for (int i = 0; i < SampleType::kN / 9; ++i, j += 3)
        {
            for (int k = 0; k < 9; ++k, ++inputIt)
            {
                image.At(x + j + k % 3, y + k / 3)[0] = *inputIt;
            }
        }
        return j;
    }

    __host__ void LiftingCodec::DrawSamples(Image1f& image, const LiftingCodec::InputSampleList& inputSamples, const LiftingCodec::OutputSampleList& targetSamples, const LiftingCodec::OutputSampleList& outputSamples) const
    {
        //Assert(inputSamples.size() == outputSamples.size() && inputSamples.size() == targetSamples.size());

        int x = 0, y = 0;
        for (int sampleIdx = 0; sampleIdx < inputSamples.size(); ++sampleIdx)
        {            
            int j = 0;
            if (!inputSamples.empty())
            {
                j += RenderSampleBlock(image, inputSamples[sampleIdx], x + j, y) + 1;
            }
            if (!targetSamples.empty())
            {
                j += RenderSampleBlock(image, targetSamples[sampleIdx], x + j, y) + 1;
            }
            if (!outputSamples.empty())
            {
                j += RenderSampleBlock(image, outputSamples[sampleIdx], x + j, y) + 1;
            }

            y += 4; 
            if (y >= image.Height() - 3)
            {
                x += j + 2;
                y = 0;
                if (x >= image.Width() - j - 2) { return; }
            }
        }
    }

    __host__ Image3f LiftingCodec::Encode(const Image3f& inputImage)
    { 
        //RunTensorTests(true);
        
        Image3f gammaImage = inputImage;

        gammaImage.Saturate();
        //gammaImage.ApplyGamma(1 / 2.2f);
            
        Image3f waveletImage = WaveletTransform(gammaImage, 1, 1);       

        /*for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            Image1f chnlData = waveletImage.ExtractChannel(chnlIdx);

            GenerateTrainingSet(chnlData);

            waveletImage.EmplaceChannel(chnlData, chnlIdx);
        }*/ 

        Image1f chnlData = waveletImage.ExtractChannel(0);

        PrepareEncoder(chnlData);

        printf_green("Training...\n");

        ////////////////////////////////////////////////////////////////////////////////////////

        printf("Generating training set...\n");
        InputSampleList inputSamples;
        OutputSampleList targetSamples, outputSamples;
        std::vector<float> inputMeans;
        GenerateTrainingSet(chnlData, 8783652, inputSamples, inputMeans, targetSamples);

        ////////////////////////////////////////////////////////////////////////////////////////

        printf("Training MLP...\n");
        NN::MLP mlp;
        mlp.Train(inputSamples, targetSamples);        

        ////////////////////////////////////////////////////////////////////////////////////////

        constexpr bool kInferImageCoeffs = false;
        constexpr bool kTestInference = true;

        if (kInferImageCoeffs)
        {
            const int kSamplesPerBatch = 10000;
            const int kNumPixels = (chnlData.Width() / 2) * (chnlData.Height() / 2);
            auto readSamples = [&](std::vector<InputSample>& samples, const int batchIdx) -> bool
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
                        const int x = pixelIdx % (chnlData.Width() / 2), y = pixelIdx / (chnlData.Width() / 2);
                        auto [sample, mean] = GenerateInputSample(x, y);
                        samples.push_back(sample);
                        inputMeans.push_back(mean);
                    }
                    return true;
                }
            };

            auto writeSamples = [&](const std::vector<OutputSample>& samples, int batchIdx) -> void
            {
                for (int sampleIdx = 0, pixelIdx = batchIdx; sampleIdx < samples.size(); ++sampleIdx, ++pixelIdx)
                {
                    auto sampleIt = samples[sampleIdx].begin();
                    for (int quadIdx = 1; quadIdx < 4; ++quadIdx)
                    {
                        const int basisU = (chnlData.Width() / 2) * (quadIdx & 1);
                        const int basisV = (chnlData.Height() / 2) * ((quadIdx >> 1) & 1);

                        const int x = pixelIdx % (chnlData.Width() / 2) + basisU;
                        const int y = pixelIdx / (chnlData.Width() / 2) + basisV;

                        *chnlData.At(x, y) = samples[sampleIdx][4 + (quadIdx - 1) * 9] *inputMeans[sampleIdx];
                        /*for (int v = -1; v <= 1; ++v)
                        {
                            for (int u = -1; u <= 1; ++u, ++sampleIt)
                            {
                                Assert(sampleIt != samples[sampleIdx].end());
                                if (chnlData.Contains(x + u, y + v))
                                {
                                    *chnlData.At(x + u, y + v) += *sampleIt * inputMeans[sampleIdx];
                                }
                            }
                        }*/
                    }
                }
            };

            // Clear the finest-scale wavelet coefficients
            ImageRect region(0, 0, waveletImage.Width() / 2, waveletImage.Height() / 2);
            chnlData.ParallelMap([&](int x, int y, int, float* pixel)
                {
                    if (!region.Contains(x, y)) { *pixel = 0; }
                });

            // Infer the coefficients
            printf("Reconstructing wavelet coefficients\n");
            mlp.Infer(readSamples, writeSamples);

            // Normalise the accumulated values
            /*chnlData.ParallelMap([&](int x, int y, int, float* pixel)
                {
                    if (!region.Contains(x, y)) { *pixel /= 9; }
                });*/
        }

        ////////////////////////////////////////////////////////////////////////////////////////

        if (kTestInference)
        {
            GenerateTrainingSet(chnlData, 235265, inputSamples, inputMeans, targetSamples);

            mlp.Infer(
                [&](std::vector<InputSample>& samples, const int batchIdx) -> bool
                {
                    if (batchIdx != 0) { return false; }
                    else
                    {
                        samples = inputSamples;
                        return true;
                    }
                },
                [&](const std::vector<OutputSample>& samples, const int batchIdx) -> void
                {
                    outputSamples = samples;
                }
                );

            DrawSamples(chnlData, inputSamples, targetSamples, outputSamples);
        }

        ////////////////////////////////////////////////////////////////////////////////////////

        waveletImage.Erase();     
        waveletImage.Resize(chnlData);
        waveletImage.ParallelMap([&](const int x, const int y, const int, float* pixel)
            {
                const float& c = chnlData.At(x, y)[0];
                pixel[(c < 0) ? 0 : 1] = std::pow(std::abs(c), 2.0f);
            }); 

        /*for (int i = 0; i < 3; ++i)
        {
            waveletImage.EmplaceChannel(chnlData, i);
        }*/

        /*waveletImage.Resize(m_heuristicImage);
        for (int i = 0; i < 3; ++i)
        {
            waveletImage.EmplaceChannel(m_heuristicImage, i);
        }*/

        return waveletImage;
    }

    __host__ Image3f LiftingCodec::Decode(const Image3f& inputImage)
    {
        Image3f waveletImage = WaveletTransform(inputImage, -1, 1);

        //waveletImage.ApplyGamma(2.2f);

        return waveletImage;
    }
}
