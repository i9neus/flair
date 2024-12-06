#pragma once

#include "core/image/Image.h"
#include "core/math/wavelets/1d/LiftingDWT1.h"
#include "core/nn/Tensor1D.cuh"
#include <array>

namespace Flair
{
	class LiftingCodec
	{
	private:
		using Sample = Tensor1D<16, false>;
		using SampleList = std::vector<Sample>;
			
		std::array<Image1f, 4>	m_mipMap;

	public:
		__host__ LiftingCodec();

		__host__ Image3f Encode(const Image3f& inputImage);
		__host__ Image3f Decode(const Image3f& inputImage);

	private:
		__host__ void PrepareEncoder(const Image1f& waveletImage);
		__host__ Image3f WaveletTransform(const Image3f& inputImage, const int direction) const;
		__host__ void GenerateTrainingSet(Image1f& inputImage, const int basisU, const int basisV, const int seed, LiftingCodec::SampleList& inputSamples, std::vector<float>& inputMeans,	LiftingCodec::SampleList& targetSamples) const;
		__host__ std::pair<LiftingCodec::Sample, float> GenerateInputSample(const int x, const int y) const;
	};
}