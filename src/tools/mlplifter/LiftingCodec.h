#pragma once

#include "core/image/Image.h"
#include "core/math/wavelets/LiftingDWT.h"

namespace Flair
{
	class LiftingCodec
	{
	public:
		LiftingCodec();

		Image3f Encode(const Image3f& inputImage) const;
		Image3f Decode(const Image3f& inputImage) const;

	private:
		Image3f Transform(const Image3f & inputImage, const int direction) const;

		void EncodeChannel(const Image1f& chnlData, Image1f& waveletData) const;

	};
}