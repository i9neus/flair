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
        return 2 * value / mean - 1;
        //return 2. * value - 1.0;
        //return value;
    }

    __host__ __inline__ float MapInverse(const float value, const float mean)
    {
        //return value * mean + mean;
        //return value + mean;
        return mean * 0.5 * (value + 1);
        //return (value + 1.0) / 2.0;
        //return value;
    }

    __host__ std::tuple<LiftingCodec::InputSample, float, float, float> LiftingCodec::GenerateInputSample(const float x, const float y, UniformDistribution* uniform) const
    {
        InputSample sample;
        float mean = 0;
        auto it = sample.begin();
        float p = 0, q = 0;
        if (uniform)
        {
            p = (*uniform)();
            q = (*uniform)();
        }

        constexpr float kScaleFactor = 0.75;
        constexpr float kOffset = 0.5;

        // First 5x5 samples contain pixel values
        for (int j = -3; j <= 3; ++j)
        {
            for (int i = -3; i <= 3; ++i, ++it)
            {
                float u = (x + i * kScaleFactor + p + kOffset) / float(m_mipMap[1].Width());
                float v = (y + j * kScaleFactor + q + kOffset) / float(m_mipMap[1].Height());
                const float f = m_mipMap[1].Sample<kImageBilinear>(u, v);
                *it = f;
                mean += f;
            }
        }

        // Normalise the samples
        it = sample.begin();
        mean = std::max(1e-3f, mean / 49);
        for (int i = 0; i < 49; ++i, ++it)
        {
            *it = MapForward(*it, mean);
        }

        // Remaining 10 samples contain positional encoding of jittered offsets
        /*for (int i = 0; i < 5; ++i)
        {
            sample[25 + i] = 0;// std::cos(kPi * float(i + 1) * (p + fract(x)));
            sample[25 + 5 + i] = 0;// std::cos(kPi * float(i + 1) * (q + fract(y)));
        }*/

        return { sample, mean, p, q };
    }

    __host__ LiftingCodec::OutputSample LiftingCodec::GenerateTargetSample(const float x, const float y, const std::tuple<InputSample, float, float, float>& inputSample) const
    {
        const float mean = std::get<1>(inputSample);
        // Jitter interval of 1 at mip level 2 correcponds to an interval of 2 at this level
        const float p = std::get<2>(inputSample) * 2;
        const float q = std::get<3>(inputSample) * 2;

        constexpr float kOffset = 0.5;

        OutputSample sample;
        auto it = sample.begin();
        for (int j = -1; j <= 1; ++j)
        {
            for (int i = -1; i <= 1; ++i, ++it)
            {
                float u = (x + i + p + kOffset) / float(m_mipMap[0].Width());
                float v = (y + j + q + kOffset) / float(m_mipMap[0].Height());
                const float f = m_mipMap[0].Sample<kImageBilinear>(u, v);
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

        float maxFeatureVal;
        //using Heuristic = DCTFeatureHeuristic<2>;
        using Heuristic = VarianceHeuristic<1>;

        Heuristic::Classify(m_mipMap[0], ImageRect(0, 0, m_mipMap[0].Width(), m_mipMap[0].Height()), m_heuristicImage, maxFeatureVal, 2);

        // Create empty histograms
        constexpr int kPDFSize = 100;
        std::vector<float> pdf(kPDFSize), cdf(kPDFSize, 0.0f);
        std::vector<std::vector<std::pair<uint16_t, uint16_t>>> histogram(kPDFSize);

        // Bucket each pixel based on its relative variance
        m_heuristicImage.Map([&](const int x, const int y, float* pixel)
            {
                //float importance = saturate(std::pow(*pixel / maxFeatureVal, 0.5f) * 2.f);  // DCT heuristic
                float importance = saturate(*pixel / maxFeatureVal);                            // Variance heuristic

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

            inline float RandReal() { return float(rng(mt)) / float(std::numeric_limits<int>::max()); }
            inline int RandInt() { return rng(mt); }
        };

        const int kNumSamples = 100000;// m_mipMap[1].Area();

        Threaded<ThreadCtx> runner(std::min(16, kNumSamples));
        runner.Initialise([&](ThreadCtx& ctx, int i, int N)
            {
                ctx.mt = std::mt19937(std::hash<int>{}(i + seed));
                ctx.rng = std::uniform_int_distribution<int>();
                ctx.uniform = UniformDistribution(0.0, 0.5, std::hash<int>{}(i * 9871 + seed));
            });

        std::atomic<int> numSamples(0);
        Threaded<ThreadCtx>::Functor sampleFunctor = [&](ThreadCtx& ctx, int i, int N)
        {
            const int startSample = kNumSamples * i / N, endSample = kNumSamples * (i + 1) / N;
            const int numThreadSamples = endSample - startSample;
            ctx.inputSamples.resize(numThreadSamples);
            ctx.inputMeans.resize(numThreadSamples);
            ctx.targetSamples.resize(numThreadSamples);

            for (int sampleIdx = 0; sampleIdx < numThreadSamples; ++sampleIdx)
            {
                std::vector<std::pair<uint16_t, uint16_t>>* bucket = nullptr;
                do
                {
                    // Draw a bucket from the density PDF
                    auto lower = std::lower_bound(cdf.begin(), cdf.end(), ctx.RandReal());
                    int bucketIdx = std::distance(cdf.begin(), lower);
                    Assert(bucketIdx < histogram.size());
                    bucket = &histogram[bucketIdx];
                } while (bucket->size() == 0);

                // Draw a sample from the bucket (coordinates in mipmap level 0)
                //auto [x, y] = (*bucket)[ctx.RandInt() % bucket->size()];
                
                const int x = ctx.RandInt() % m_mipMap[0].Width();
                const int y = ctx.RandInt() % m_mipMap[0].Height();

                //const int x = 2 * ((sampleIdx + startSample) % m_mipMap[1].Width()) + (ctx.RandInt() % 2);
                //const int y = 2 * ((sampleIdx + startSample) / m_mipMap[1].Width()) + (ctx.RandInt() % 2);

                // Construct the input sample
                auto inputSample = GenerateInputSample(x * 0.5, y * 0.5, &ctx.uniform);
                ctx.inputSamples[sampleIdx] = std::get<0>(inputSample);
                ctx.inputMeans[sampleIdx] = std::get<1>(inputSample);

                // ...and the target sample
                ctx.targetSamples[sampleIdx] = GenerateTargetSample(x, y, inputSample);
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
    __host__ int RenderSampleBlock(Image1f& image, const SampleType& sample, int x, int y, int size)
    {
        auto inputIt = sample.begin();
        for (int k = 0; k < SampleType::kN; ++k, ++inputIt)
        {
            image.At(x + k % size, y + k / size)[0] = *inputIt;
        }
        return size;
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
                j += RenderSampleBlock(image, inputSamples[sampleIdx], x + j, y, 7) + 1;
            }
            if (!targetSamples.empty())
            {
                j += RenderSampleBlock(image, targetSamples[sampleIdx], x + j, y, 3) + 1;
            }
            if (!outputSamples.empty())
            {
                j += RenderSampleBlock(image, outputSamples[sampleIdx], x + j, y, 3) + 1;
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
        constexpr bool kInferImageCoeffs = true;
        constexpr bool kCollaborative = false;
        constexpr bool kTestInference = false;
        constexpr bool kSignedColourView = false;
        
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

        GenerateTrainingSet(8783652, inputSamples, inputMeans, targetSamples);
        //SerialiseTrainingSet(inputSamples, targetSamples);

        //DeserialiseTrainingSet(inputSamples, targetSamples);

        //inputSamples.resize(1);
        //targetSamples.resize(1);

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
            const int kNumPixels = m_mipMap[0].Area();
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
                        const int x = pixelIdx % m_mipMap[0].Width(), y = pixelIdx / m_mipMap[0].Width();
                        auto [sample, mean, p, q] = GenerateInputSample(x * 0.5, y * 0.5, nullptr);
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
                    const int x = pixelIdx % m_mipMap[0].Width();
                    const int y = pixelIdx / m_mipMap[0].Width();

                    if (!kCollaborative)
                    {
                        *chnlData.At(x, y) = MapInverse(samples[sampleIdx][12], inputMeans[sampleIdx]);
                    }
                    else
                    {
                        for (int v = -1, i = 0; v <= 1; ++v)
                        {
                            for (int u = -1; u <= 1; ++u, ++i)
                            {
                                if (chnlData.Contains(x + u, y + v))
                                {
                                    //const float norm = ((u == 0 && v == 0) ? 8 : 1) / 16.f;
                                    const float norm = 1 / 9.f;
                                    *chnlData.At(x + u, y + v) += MapInverse(samples[sampleIdx][i], inputMeans[sampleIdx]) * norm;
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
