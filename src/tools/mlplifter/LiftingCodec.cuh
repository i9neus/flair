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
		using InputSample = Tensor1D<36>;
		using OutputSample = Tensor1D<27>;
		using InputSampleList = std::vector<InputSample>;
		using OutputSampleList = std::vector<OutputSample>;
			
		std::array<Image1f, 2>	m_mipMap;
		Image1f m_heuristicImage;

	public:
		__host__ LiftingCodec();

		__host__ Image3f Encode(const Image3f& inputImage);
		__host__ Image3f Decode(const Image3f& inputImage);

	private:
		__host__ void PrepareEncoder(const Image1f& waveletImage);
		__host__ void GenerateTrainingSet(Image1f& inputImage, const int seed, InputSampleList& inputSamples, std::vector<float>& inputMeans, OutputSampleList& targetSamples);
		__host__ std::pair<InputSample, float> GenerateInputSample(const int x, const int y) const;
		__host__ OutputSample GenerateTargetSample(const int x, const int y) const;
		__host__ float CoefficientAt(float u, float v, const int quadX, const int quadY, const int mipLevel) const;
		__host__ void DrawSamples(Image1f& image, const InputSampleList& inputSample, const OutputSampleList& targetSamples, const OutputSampleList& outputSamples) const;

	};
}