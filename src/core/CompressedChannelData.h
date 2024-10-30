#pragma once

#include "Includes.h"
#include "coder/ArithmeticCoder.h"
#include "coder/RLECoder.h"
#include "ByteStream.h"
#include "Half.h"

namespace HDRI
{	
	namespace MagicNumbers
	{
		static constexpr MagicType kChannelHeader = 0x328ab01c;
		static constexpr MagicType kPrecintHeader = 0x176a77bb;
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
			MagicType									magic = MagicNumbers::kChannelHeader;
			uint8_t										numPrecincts = 0;
			uint8_t										minCompressedPrecinct = 0;
			int											sizeUncompressedPrecinctData = 0;
			int											sizeDwtPassNorms = 0;
		};

		// Stored at the beginning of each precinct in each channel
		struct StreamPrecinctBlockHeader
		{
			MagicType									magic = MagicNumbers::kPrecintHeader;
			uint8_t										flags = 0;
			uint16_t									sizePrecinctModelPMFTable = 0;  // Number of entries in the model PMF table
			int											sizeCompressedPrecinct = 0;		// Size of the compressed precinct
			int											sizeRLEBlockTable = 0;
		};

		struct CompressedPrecinctData
		{
			StreamPrecinctBlockHeader					header;
			std::vector<uint8_t>						compressedData;
			PrecinctModel								arithModel;
			RLECoder<uint16_t>::BlockTable				rleBlockTable;

			size_t SizeOf() const
			{
				size_t size = 0;
				size += sizeof(header);
				size += sizeof(uint8_t) * compressedData.size();
				size += sizeof(PrecinctModel::value_type) * arithModel.size();
				size += sizeof(RLECoder<uint16_t>::BlockTable::value_type) * rleBlockTable.size();

				return size;
			}
		};

	public:
		CompressedChannelData() = default;
		
		StreamChannelHeader								header;		
		std::vector<float>								uncompressedPrecinctData;		
		std::vector<CompressedPrecinctData>				compressedPrecinctData;

		std::vector<int>								quantRates;
		std::vector<float>								dwtPassNorms;

	public:
		void Serialise(ByteStream& stream, const int chnlIdx)
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
				SerialisePrecinct(compressedPrecinctData[i], stream);
			}
		}

		void Deserialise(ByteStream& stream, const int chnlIdx)
		{
			// Deserialise and check the header
			stream >> header;
			AssertMsg(header.magic == MagicNumbers::kChannelHeader, "Corrupt byte stream: mMagic number mismatch in channel data header.");
			AssertMsg(header.sizeUncompressedPrecinctData > 0, "Corrupt byte stream: header.sizeUncompressedPrecinctData < 0");
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
				DeserialisePrecinct(compressedPrecinctData[i], stream);
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

	private:
		void SerialisePrecinct(CompressedPrecinctData& precinct, ByteStream& stream) const
		{
			// Serialise the precinct header
			precinct.header.magic = MagicNumbers::kPrecintHeader;
			precinct.header.sizePrecinctModelPMFTable = precinct.arithModel.size();
			precinct.header.sizeCompressedPrecinct = precinct.compressedData.size();
			precinct.header.sizeRLEBlockTable = precinct.rleBlockTable.size();
			stream << precinct.header;

			// Serialise the coder
			for (const auto& entry : precinct.arithModel)
			{
				stream << entry.first << entry.second;
			}

			// Serialise the RLE block table
			stream << precinct.rleBlockTable;

			// Serialise the encoded data
			stream << precinct.compressedData;
		}

		void DeserialisePrecinct(CompressedPrecinctData& precinct, ByteStream& stream)
		{
			// Deseriaise the precinct header
			stream >> precinct.header;
			AssertMsg(precinct.header.magic == MagicNumbers::kPrecintHeader, "Corrupt byte stream: magic number mismatch in precinct data header.");

			// Deserialise the coder
			precinct.arithModel.resize(precinct.header.sizePrecinctModelPMFTable);
			for (const auto& entry : precinct.arithModel)
			{
				stream.Read(entry.first);
				stream.Read(entry.second);
			}
			
			// Deserialise the RLE block table
			stream.Read(precinct.rleBlockTable, precinct.header.sizeRLEBlockTable);

			// Deserialise the encoded data
			stream.Read(precinct.compressedData, precinct.header.sizeCompressedPrecinct);
		}
	};
}