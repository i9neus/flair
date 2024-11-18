#include "LiftingCodec.h"

#include "core/math/wavelets/cuda/LiftingMLP.cuh"
#include "core/math/wavelets/StaticDWT.h"
#include "core/math/wavelets/LiftingDWT.h"

namespace Flair
{
	LiftingCodec::LiftingCodec()
	{
	}

    Image3f LiftingCodec::Transform(const Image3f& inputImage, const int direction) const
    {
        Image3f outputImage(inputImage.Width(), inputImage.Height());

        // Functor that extracts a channel from the image, encodes it, then emplaces any generated wavelet coefficients
        std::function<void(int)> EncodeChannelFunctor = [&](int chnlIdx)
        {
            Image1f waveletData = inputImage.ExtractChannel(chnlIdx);

            // In-place transform the image using the DCT
            LiftingDWT<float> dwt(waveletData.Width());
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

    Image3f LiftingCodec::Encode(const Image3f& inputImage) const 
    { 
        Image3f gammaImage = inputImage;

        gammaImage.Saturate();
        gammaImage.ApplyGamma(1 / 2.2f);
            
        return Transform(gammaImage, 1); 
    }

    Image3f LiftingCodec::Decode(const Image3f& inputImage) const 
    { 
        Image3f outputImage = Transform(inputImage, -1);

        outputImage.ApplyGamma(2.2f);

        return outputImage;
    }
}
