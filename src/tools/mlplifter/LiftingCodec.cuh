#pragma once

#include "core/image/Image.h"
#include "core/math/wavelets/1d/LiftingDWT1.h"
#include "core/nn/Tensor1D.cuh"
#include "core/nn/DataLoader.cuh"

namespace Flair
{
	class MLPDataset : public NN::DataLoader<Tensor1D<16, false>>
	{
	public:
		using SampleList = std::vector<Sample>;

		std::vector<Sample> inputSamples;
		std::vector<Sample> targetSamples;

	public:
		__host__ MLPDataset() = default;

		__host__ virtual size_t Size() const override final { return inputSamples.size(); }

		__host__ virtual std::pair<const std::vector<Sample>*, const std::vector<Sample>*> Data() const override final
		{ 
			 Assert(!inputSamples.empty() && !targetSamples.empty()); 
			 return { &inputSamples, &targetSamples };
		}
	};
	
	class LiftingCodec
	{
	public:
		__host__ LiftingCodec();

		__host__ Image3f Encode(const Image3f& inputImage) const;
		__host__ Image3f Decode(const Image3f& inputImage) const;

	private:
		__host__ Image3f WaveletTransform(const Image3f& inputImage, const int direction) const;
		__host__ MLPDataset GenerateTrainingSet(Image1f& inputImage) const;
	};
}