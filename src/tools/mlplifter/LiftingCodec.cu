#include "LiftingCodec.cuh"

#include "core/nn/mlp/MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/math/wavelets/2d/StaticDWT2.h"
#include "core/utils/ConsoleUtils.h"
#include "core/utils/ThreadUtils.h"
#include "core/io/IOUtils.h"
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

    __host__ __inline__ float MapForward(const float value, const float mean)
    {
        //return (value - mean) / mean;
        //return value - mean;
        return 2. * value - 1.0;
        //return value;
    }

    __host__ __inline__ float MapInverse(const float value, const float mean)
    {
        //return value * mean + mean;
        //return value + mean;
        return (value + 1.0) / 2.0;
        //return value;
    }

    __host__ std::tuple<LiftingCodec::InputSample, float> LiftingCodec::GenerateInputSample(const int x, const int y) const
    {
        InputSample sample;
        float mean = 0;
        auto it = sample.begin();   

        // First 5x5 samples contain pixel values
        for (int j = -2; j < 4; ++j)
        {
            for (int i = -2; i < 4; ++i, ++it)
            {
                const float f = m_mipMap[1].Sample(x + i, y + j);
                //const float f = float(std::sqrt(float(i * i) + float(j * j)) <= 2);
                *it = f;
                mean += f;
            }
        }
        
        // Normalise the samples
        it = sample.begin();
        mean = std::max(1e-3f, mean / 36);
        for (int i = 0; i < 36; ++i, ++it)
        {
            *it = MapForward(*it, mean);
        }      

        return { sample, mean };
    }

    __host__ LiftingCodec::OutputSample LiftingCodec::GenerateTargetSample(const int x, const int y, const std::tuple<InputSample, float>& inputSample) const
    {
        const float mean = std::get<1>(inputSample);

        OutputSample sample;
        auto it = sample.begin();
        for (int j = -2; j < 4; ++j)
        {
            for (int i = -2; i < 4; ++i, ++it)
            {
                const float f = m_mipMap[0].Sample(x + i, y + j);
                //const float f = float(std::sqrt(float(i * i) + float(j * j)) <= 2);
                *it = MapForward(f, mean);
            }
        }

        return sample;
    }

    __host__ void LiftingCodec::GenerateTrainingSet(const int seed, LiftingCodec::InputSampleList& inputSamples, std::vector<float>& inputMeans, LiftingCodec::OutputSampleList& targetSamples)
    {
        inputSamples.clear();
        inputMeans.clear();
        targetSamples.clear();        

        //using Heuristic = DCTFeatureHeuristic<5>;
        using Heuristic = VarianceHeuristic<7>;

        float maxFeatureVal;
        Heuristic::Classify(m_mipMap[1], ImageRect(0, 0, m_mipMap[1].Width(), m_mipMap[1].Height()), m_heuristicImage, maxFeatureVal);

        // Create empty histograms
        constexpr int kPDFSize = 100;
        std::vector<float> pdf(kPDFSize), cdf(kPDFSize, 0.0f);
        std::vector<std::vector<std::pair<uint16_t, uint16_t>>> histogram(kPDFSize);

        // Bucket each pixel based on its relative variance
        m_heuristicImage.Map([&](const int x, const int y, float* pixel)
            {
                //float importance = saturate(std::pow(*pixel / maxFeatureVal, 0.4f) * 2.f);
                float importance = saturate(std::pow(*pixel / maxFeatureVal, 1.f));
                const int bucketIdx = clamp(int(kPDFSize * importance), 0, kPDFSize - 1);
                histogram[bucketIdx].emplace_back(uint16_t(x), uint16_t(y));
            });

        // Construct and normalise a PDF and CDF based on the inverse bucket size
        constexpr float kMaxPdf = 100.f;
        for (int i = 0; i < kPDFSize; ++i)
        {
            pdf[i] = std::min(kMaxPdf, float(m_heuristicImage.Area()) / std::max(1ull, histogram[i].size()));
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
            InputSampleList                     inputSamples;
            OutputSampleList                    targetSamples;
            std::vector<float>                  inputMeans;
            std::mt19937                        mt;
            std::uniform_int_distribution<int>  rng;
            UniformDistribution                 uniform;

            inline float RandReal()             { return float(rng(mt)) / float(std::numeric_limits<int>::max()); }
            inline int RandInt()                { return rng(mt); }
        };

        //const int kNumSamples = sqr(m_mipMap[1].Width());
        constexpr int kNumSamples = 1024;

        Threaded<ThreadCtx> runner(std::min(16, kNumSamples));
        runner.Initialise([&](ThreadCtx& ctx, int i, int N)
            {
                ctx.mt = std::mt19937(std::hash<int>{}(i + seed));
                ctx.rng = std::uniform_int_distribution<int>();
                ctx.uniform = UniformDistribution(0.0, 0.5, std::hash<int>{}(i * 9871 + seed));
            });

        Threaded<ThreadCtx>::Functor sampleFunctor = [&](ThreadCtx& ctx, int i, int N)
        {
            const int numThreadSamples = (kNumSamples * (i + 1) / N) - (kNumSamples * (i) / N);
            ctx.inputSamples.resize(numThreadSamples);
            ctx.inputMeans.resize(numThreadSamples);
            ctx.targetSamples.resize(numThreadSamples);

            for (int sampleIdx = 0, j = kNumSamples * i / N; sampleIdx < numThreadSamples; ++sampleIdx, ++j)
            {
                std::vector<std::pair<uint16_t, uint16_t>>* bucket = nullptr;
                do
                {
                    // Draw a bucket from the density PDF
                    auto lower = std::lower_bound(cdf.begin(), cdf.end(), ctx.RandReal());
                    int bucketIdx = std::distance(cdf.begin(), lower);
                    Assert(bucketIdx < histogram.size());
                    bucket = &histogram[bucketIdx];
                } 
                while (bucket->size() == 0);

                // Draw a sample from the bucket (coordinates in mipmap level 0)
                auto [x, y] = (*bucket)[ctx.RandInt() % bucket->size()];
                
                // Random sample the image
                //const int x = ctx.RandInt() % m_mipMap[1].Width();
                //const int y = ctx.RandInt() % m_mipMap[1].Height();
                
                // Sequentially sample the image
                //const int x = j % m_mipMap[1].Width();
                //const int y = j / m_mipMap[1].Width();

                // Construct the input sample
                auto inputSample = GenerateInputSample(x, y);
                ctx.inputSamples[sampleIdx] = std::get<0>(inputSample);
                ctx.inputMeans[sampleIdx] = std::get<1>(inputSample);

                // ...and the target sample
                ctx.targetSamples[sampleIdx] = GenerateTargetSample(x * 2, y * 2, inputSample);
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

    template<int Size>
    __host__ int RenderSampleBlock(Image1f& image, const Tensor1D<Size>& sample, int x, int y)
    {
        const int kEdge = std::sqrt(Size);
        auto inputIt = sample.begin();
        for (int k = 0; k < Size; ++k, ++inputIt)
        {
            image.At(x + k % kEdge, y + k / kEdge)[0] = *inputIt;
        }
        return 8;
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

            y += 8; 
            if (y >= image.Height() - 8)
            {
                x += j + 2;
                y = 0;
                if (x >= image.Width() - j - 8) { return; }
            }
        }
    }

    __host__ void LiftingCodec::SerialiseTrainingSet(const LiftingCodec::InputSampleList& inputSamples, const LiftingCodec::OutputSampleList& targetSamples) const
    {
        const char* kFilePath = "C:/projects/probenet/src/experiments/flair/dataset.dat";
        std::ofstream file(kFilePath, std::ios::binary);
        Assert(file.is_open());
        
        for (int i = 0; i < inputSamples.size(); ++i)
        {
            file.write(reinterpret_cast<const char*>(&inputSamples[i][0]), sizeof(float) * InputSample::kN);
            file.write(reinterpret_cast<const char*>(&targetSamples[i][0]), sizeof(float) * OutputSample::kN);
        }
        
        printf_green("Wrote %i sample pairs to '%s'!\n", inputSamples.size(), kFilePath);
        file.close();
    }

    __host__ void LiftingCodec::DeserialiseTrainingSet(LiftingCodec::InputSampleList& inputSamples, LiftingCodec::OutputSampleList& targetSamples) const
    {
        const char* kFilePath = "C:/projects/probenet/src/experiments/flair/dataset.dat";
        std::vector<float> rawData;
        Assert(IO::DeserialiseArray(rawData, kFilePath) != 0);
        Assert(rawData.size() % (InputSample::kN + OutputSample::kN) == 0);

        const int kNumSamples = rawData.size() / (InputSample::kN + OutputSample::kN);
        inputSamples.resize(kNumSamples);
        targetSamples.resize(kNumSamples);

        for (int sampleIdx = 0, rawIdx = 0; sampleIdx < kNumSamples; ++sampleIdx)
        {
            std::memcpy(inputSamples[sampleIdx].Data(), &rawData[rawIdx], sizeof(float) * InputSample::kN);
            rawIdx += InputSample::kN;
            std::memcpy(targetSamples[sampleIdx].Data(), &rawData[rawIdx], sizeof(float) * OutputSample::kN);
            rawIdx += OutputSample::kN;
        }

        printf_green("Loaded %i samples from '%s'!\n", inputSamples.size(), kFilePath);
    }

    __host__ LiftingCodec::OutputSampleList LiftingCodec::DeserialiseInferenceDataset() const
    {
        const char* kFilePath = "C:/projects/probenet/src/experiments/flair/inference.dat";
        std::ifstream file(kFilePath, std::ios::in | std::ios::binary);
        Assert(file.is_open());

        file.seekg(0, std::ios::end);
        const int fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        Assert(fileSize % sizeof(float) == 0);
        Assert((fileSize / sizeof(float)) % OutputSample::kN == 0);

        OutputSampleList samples;
        samples.resize(fileSize / (sizeof(float) * OutputSample::kN));         
        file.read(reinterpret_cast<char*>(samples.data()), fileSize);

        return samples;
    }

    __host__ Image3f LiftingCodec::Encode(const Image3f& inputImage)
    { 
        constexpr bool kTrainMLP = true;
        constexpr bool kShowPytorchRef = false;
        constexpr bool kInferImageCoeffs = false;
        constexpr bool kCollaborative = false;
        constexpr bool kTestInference = true;
        constexpr bool kSignedColourView = true;
        
        RunTensorTests(false);
        //return inputImage;
        
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

        // Create a two-level mipmap
        m_mipMap[0] = gammaImage.ExtractChannel(0);
        m_mipMap[1] = Crop(chnlData, ImageRect(0, 0, waveletImage.Width() / 2, waveletImage.Height() / 2));

        printf_green("Training...\n");

        ////////////////////////////////////////////////////////////////////////////////////////

        printf("Generating training set...\n");
        InputSampleList inputSamples;
        OutputSampleList targetSamples, outputSamples;
        std::vector<float> inputMeans;
        //GenerateTrainingSet(8783652, inputSamples, inputMeans, targetSamples);

        //inputSamples.resize(1);
        //targetSamples.resize(1);

        //SerialiseTrainingSet(inputSamples, targetSamples);
        DeserialiseTrainingSet(inputSamples, targetSamples);

        //for (auto& f : inputSamples) { f = 1; }
        //for (auto& f : targetSamples) { f = 0.5f; }

        ////////////////////////////////////////////////////////////////////////////////////////

        printf("Training MLP...\n");
        NN::MLP mlp;
        if (kTrainMLP)
        {
            mlp.Train(inputSamples, targetSamples);
        }

        ////////////////////////////////////////////////////////////////////////////////////////        

        if (kInferImageCoeffs)
        {
            const int kSamplesPerBatch = 10000;
            const int kNumPixels = m_mipMap[1].Area();
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
                        const int x = pixelIdx % m_mipMap[1].Width(), y = pixelIdx / m_mipMap[1].Width();
                        auto [sample, mean] = GenerateInputSample(x, y);
                        samples.push_back(sample);
                        inputMeans.push_back(mean);
                    }
                    return true;
                }
            };

            auto writeSamples = [&](const std::vector<OutputSample>& samples, int batchIdx) -> void
            {
                auto sample = samples.cbegin();
                for (int sampleIdx = 0, pixelIdx = batchIdx; sampleIdx < samples.size(); ++sampleIdx, ++pixelIdx, ++sample)
                {
                     const int x = pixelIdx % m_mipMap[1].Width(), y = pixelIdx / m_mipMap[1].Width();

                    if (!kCollaborative)
                    {
                        for (int v = 0; v < 2; ++v)
                        {
                            for (int u = 0; u < 2; ++u)
                            {
                                *chnlData.At(x*2+u, y*2+v) = MapInverse((*sample)[(2+v)*6 + (2+u)], inputMeans[sampleIdx]);
                            }
                        }                        
                    }
                    else
                    {
                        for (int v = -2, i = 0; v < 4; ++v)
                        {
                            for (int u = -2; u < 4; ++u, ++i)
                            {
                                if (chnlData.Contains(x*2 + u, y*2 + v))
                                {
                                    *chnlData.At(x*2+u, y*2+v) += MapInverse((*sample)[(2+v)*6 + (2+u)], inputMeans[sampleIdx]);
                                }
                            }
                        }
                    }
                }
               
            };

            // Clear the finest-scale wavelet coefficients
            chnlData.Erase();

            // Infer the coefficients
            printf("Reconstructing wavelet coefficients\n");
            mlp.Infer(readSamples, writeSamples);

            if (kCollaborative)
            {
                // Normalise the accumulated values
                chnlData.ParallelMap([&](int x, int y, int, float* pixel) { *pixel /= 9; });
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////

        if (kShowPytorchRef)
        {
            outputSamples = DeserialiseInferenceDataset();
            DrawSamples(chnlData, inputSamples, targetSamples, outputSamples);
        }

        ////////////////////////////////////////////////////////////////////////////////////////

        if (kTestInference)
        {
            //GenerateTrainingSet(235265, inputSamples, inputMeans, targetSamples);

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

        if (kSignedColourView)
        {
            waveletImage.ParallelMap([&](const int x, const int y, const int, float* pixel)
                {
                    const float& c = chnlData.At(x, y)[0];
                    pixel[(c < 0) ? 0 : 1] = std::pow(std::abs(c), 2.0f);
                });
        }
        else
        {
            //chnlData.ApplyGamma(2.2f);
            for (int i = 0; i < 3; ++i)
            {
                waveletImage.EmplaceChannel(chnlData, i);
            }
        }

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
