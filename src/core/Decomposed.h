#pragma once

#include "Includes.h"
#include "coder/ArithmeticCoder.h"

namespace HDRI
{
	struct DecomposedChannelData
	{
		using PrecinctModel = ArithmeticCoder<uint16_t>::Model;
		
		int												numPrecincts;
		int												minCompressedPrecinct;
		std::vector<float>								uncompressedPrecinctData;
		std::vector<std::vector<uint8_t>>				compressedPrecincts;
		std::vector<PrecinctModel>						precinctModels;
		std::vector<int>								quantRatesY;
		std::vector<int>								quantRatesUV;

		size_t SizeOf() const
		{
			size_t size = sizeof(minCompressedPrecinct);
			size += uncompressedPrecinctData.size() * sizeof(float);
			for (const auto& precinct : compressedPrecincts) { size += precinct.size() * sizeof(uint8_t); }
			for (const auto& model : precinctModels) { size += ::SizeOf(model); }

			return size;
		}
	};

	struct DecomposedImage
	{
		int												width;
		int												height;
		static constexpr int    						kChannels = 3;
		std::array<DecomposedChannelData, kChannels>    channelData;

		size_t SizeOf() const
		{
			size_t size = sizeof(width) + sizeof(height);
			for (const auto& data : channelData) { size += data.SizeOf(); }

			return size;
		}
	};
}