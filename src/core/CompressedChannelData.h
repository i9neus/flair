#pragma once

#include "CompressedPrecinctData.h"

namespace Flair
{	
	namespace MagicNumbers
	{
		static constexpr MagicType kChannelHeader = 0x328ab01c;
	};
	
	struct CompressedChannelData
	{
	public:
		using PrecinctModel = ArithmeticCoder<uint16_t>::Model;
		enum Flags : uint8_t { kPrecinctEmpty = 1 };

	private:
		// Stored at the beginning of each image channel
		struct StreamChannelHeader
		{
			StreamChannelHeader()
			{
				std::memset(this, 0, sizeof(StreamChannelHeader));
				magic = MagicNumbers::kChannelHeader;
			}
			
			MagicType									magic;
			uint8_t										numPrecincts;
			uint8_t										minCompressedPrecinct;
			int											sizeUncompressedPrecinctData;
			int											sizeDwtPassNorms;
		};

	public:
		CompressedChannelData() = default;
		
		StreamChannelHeader								header;		
		std::vector<float>								uncompressedPrecinctData;		
		std::vector<CompressedPrecinctData>				compressedPrecinctData;

		std::vector<int>								quantRates;
		std::vector<float>								dwtPassNorms;

	public:
		void Serialise(OutputStream& stream, const int chnlIdx)
		{
			// Reduce the uncompressed precinct data to half precision
			std::vector<uint16_t> halfBuffer;
			halfBuffer.reserve(uncompressedPrecinctData.size());
			for (const auto& f : uncompressedPrecinctData)
			{
				halfBuffer.push_back(FloatToHalfBits(f));
			}

			// Serialise the channel header
			header.sizeUncompressedPrecinctData = halfBuffer.size();
			header.sizeDwtPassNorms = dwtPassNorms.size();
			stream << header;			

			// Serialise the uncompressed precinct data, quantisation rates, DWT pass norms
			stream << halfBuffer;
			stream << quantRates;
			stream << dwtPassNorms;

			// Serialise each precinct
			for (int i = header.minCompressedPrecinct; i < compressedPrecinctData.size(); ++i)
			{
				compressedPrecinctData[i].Serialise(stream);
			}
		}

		void Deserialise(InputStream& stream, const int chnlIdx)
		{
			// Deserialise and check the header
			stream >> header;
			AssertFmt(header.magic == MagicNumbers::kChannelHeader, "Corrupt byte stream: mMagic number mismatch in channel data header.");
			AssertFmt(header.sizeUncompressedPrecinctData > 0, "Corrupt byte stream: header.sizeUncompressedPrecinctData < 0");
			compressedPrecinctData.resize(header.numPrecincts);

			// Deserialise the half-precision uncompressed precinct data and convert it back to full-precision 32-bit floats
			std::vector<uint16_t> halfBuffer;
			stream.Read(halfBuffer, header.sizeUncompressedPrecinctData);
			uncompressedPrecinctData.clear();
			uncompressedPrecinctData.reserve(header.sizeUncompressedPrecinctData);
			for (const auto& i : halfBuffer)
			{
				uncompressedPrecinctData.push_back(HalfBitsToFloat(i));
			}

			// Deserialise the quantisation rates and DWT norms
			stream.Read(quantRates, header.numPrecincts);
			stream.Read(dwtPassNorms, header.sizeDwtPassNorms);

			// Deserialise each precinct
			for (int i = header.minCompressedPrecinct; i < compressedPrecinctData.size(); ++i)
			{
				compressedPrecinctData[i].Deserialise(stream);
			}
		}

		size_t SizeOf() const
		{
			size_t size = 0;
			size += sizeof(header.minCompressedPrecinct);
			size += uncompressedPrecinctData.size() * sizeof(float) / 2; // Divide by two for half-precision float
			for (const auto& precinct : compressedPrecinctData) { size += precinct.SizeOf(); }

			return size;
		}		
	};
}