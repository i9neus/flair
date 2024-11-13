#pragma once

#include "core/image/Image.h"
#include "core/math/wavelets/cuda/LiftingDWT.cuh"

namespace Flair
{
	class LiftingCodec
	{
	public:
		LiftingCodec();

		Image3f Encode(const Image3f& inputImage);
		Image3f Decode(const Image3f& inputImage);

	private:
		void EncodeChannel(const Image1f& chnlData, Image1f& waveletData) const;

	};
}