#pragma once

#include "core/image/Image.h"
#include "core/math/wavelets/1d/LiftingDWT1.h"
#include "core/nn/Tensor1D.cuh"
#include "core/nn/DataLoader.cuh"

namespace Flair
{
	class MLPDataset : public NN::DataLoader
	{
	public:
		using Sample = std::array<float, 16>;
		using SampleList = std::vector<Sample>;

		std::vector<Sample> inputSamples;
		std::vector<Sample> targetSamples;

	public:
		__host__ MLPDataset() = default;

		__host__ virtual size_t Size() const override final { return inputSamples.size(); }
		
		__host__ virtual std::pair<float*, float*> operator[](const int idx) override final
		{
			return { inputSamples[idx].data(), targetSamples[idx].data() };
		}

		__host__ virtual std::pair<const float*, const float*> Data() const override final 
		{ 
			 Assert(!inputSamples.empty() && !targetSamples.empty()); 
			 return { inputSamples.front().data(), targetSamples.front().data() };
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