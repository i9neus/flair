#include "Analysis.h"
#include "TestIterator.h"

#define STB_IMAGE_IMPLEMENTATION
#include "External/Stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/Stb/stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include "External/TinyEXR/tinyexr.h"

namespace Flair
{
    namespace Analysis
    {
        Image3f LoadEXR(const char* inputPath)
        {
            std::printf("Loading '%s'...\n", inputPath);
            const char* err = nullptr;
            float* exrDataIn;
            int exrWidth, exrHeight;
            if (::LoadEXR(&exrDataIn, &exrWidth, &exrHeight, inputPath, &err) != TINYEXR_SUCCESS)
            {
                if (err)
                {
                    std::printf("Error: EXR reader returned error '%s'", err);
                    FreeEXRErrorMessage(err);
                }
                Assert(false);
            }

            AssertFmt(exrWidth == exrHeight, "Input image aspect ratio %i x %i is not 1:1.", exrWidth, exrHeight);            

            Flair::Image3f image(exrWidth, exrHeight);
            image.Populate([&](const int x, const int y, float* pixel)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        pixel[c] = exrDataIn[(y * exrWidth + x) * 4 + c];
                    }
                });
            delete[] exrDataIn;

            return image;
        }

        void Tune()
        {
            const char* kImagePath = "C:/Unity/SyntheticGS/Assets/HDRI/brown_photostudio_02_4k_regular.exr";

            const Image3f inputImage = LoadEXR(kImagePath);
            TestIterator iterator(inputImage);

            std::ofstream dataFile("C:/Unity/SyntheticGS/Assets/HDRI/analysis.dat");

            constexpr int kNSubdivs = 50;

            TestIterator::Functor onInit = [&](TestIterator::IterationData& iter)
            {
                auto params = iter.codec.GetEncoderParams();
                params.quantQuality = 0.0;
                params.quantAttenuation = mix(4.f, 2.f, float(iter.i) / float(iter.N));
                iter.codec.SetEncoderParams(params);
            };

            TestIterator::Functor onComplete = [&](TestIterator::IterationData& iter)
            {
                auto params = iter.codec.GetEncoderParams();
                dataFile << tfm::format("%.10f %.10f %.10f ", iter.stats.bitsPerPixel, iter.stats.ssim, params.quantQuality);
            };

            iterator.Run(kNSubdivs, onInit, onComplete);
        }

        void TuneMultiparam()
        {
            const char* kImagePath = "C:/Unity/SyntheticGS/Assets/HDRI/brown_photostudio_02_4k_regular.exr";
            
            const Image3f inputImage = LoadEXR(kImagePath);
            TestIterator iterator(inputImage);

            std::ofstream dataFile("C:/Unity/SyntheticGS/Assets/HDRI/analysis.dat");

            constexpr int kXSubdivs = 20;
            constexpr int kYSubsivs = 20;

            TestIterator::Functor onInit = [&](TestIterator::IterationData& iter)
            {
                auto params = iter.codec.GetEncoderParams();
                //params.quality = mix(0.f, 0.8f, float(iterIdx) / float(numIters));
                params.quantQuality = mix(0.f, 0.35f, float(iter.i % kXSubdivs) / kXSubdivs);
                params.quantAttenuation = mix(4.0f, 1.0f, float(iter.i / kXSubdivs) / kYSubsivs);
                iter.codec.SetEncoderParams(params);
            };

            TestIterator::Functor onComplete = [&](TestIterator::IterationData& iter)
            {
                auto params = iter.codec.GetEncoderParams();
                dataFile << tfm::format("%.10f %.10f %.10f %.10f", params.quantQuality, params.quantAttenuation, iter.stats.ssim, iter.stats.bitsPerPixel);
            };

            iterator.Run(kXSubdivs * kYSubsivs, onInit, onComplete);
        }
    }
}