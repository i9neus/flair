#include "LiftingCodec.h"

#include "core/math/wavelets/cuda/LiftingDWT.cuh"
#include "core/math/wavelets/DWT.h"

namespace Flair
{
	LiftingCodec::LiftingCodec()
	{
	}

    void LiftingCodec::EncodeChannel(const Image1f& chnlData, Image1f& waveletData) const
    {
        waveletData = chnlData;
        
        // In-place transform the image using the DCT
        StaticDWT<CDF97<float>> dwt(chnlData.Width());
        dwt.Forward(waveletData.Vector());
    }

	Image3f LiftingCodec::Encode(const Image3f& inputImage)
	{
        Image3f outputImage(inputImage.Width(), inputImage.Height());
        
        // Functor that extracts a channel from the image, encodes it, then emplaces any generated wavelet coefficients
        std::function<void(int)> EncodeChannelFunctor = [&, this](int chnlIdx)
        {
            Image1f chnlData = inputImage.ExtractChannel(chnlIdx);
            Image1f waveletData(chnlData.Width(), chnlData.Height());

            EncodeChannel(chnlData, waveletData);

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

	Image3f LiftingCodec::Decode(const Image3f& inputImage)
	{
        return Image3f();
	}
}
