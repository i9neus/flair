#pragma once

#include "Includes.h"
#include "coders/ArithmeticCoder.h"
#include "coders/RLECoder.h"
#include "io/InputStream.h"
#include "io/OutputStream.h"
#include "math/Half.h"

namespace Flair
{	
	namespace MagicNumbers
	{
		static constexpr MagicType kPrecintHeader = 0x176a77bb;
	};

	struct CompressedPrecinctData
	{
	private:
		// Stored at the beginning of each precinct in each channel
		struct StreamPrecinctBlockHeader
		{
			StreamPrecinctBlockHeader()
			{
				std::memset(this, 0, sizeof(StreamPrecinctBlockHeader));
				magic = MagicNumbers::kPrecintHeader;
			}
			
			MagicType									magic;
			uint8_t										flags;
			uint16_t									sizePrecinctModelPMFTable;  // Number of entries in the model PMF table
			int											sizeCompressedPrecinct;		// Size of the compressed precinct
			int											sizeRLEBlockTable;			// Size of the run-length encoder block table
		};

	public:		
		using PrecinctModel = ArithmeticCoder<uint16_t>::Model;
		enum Flags : uint8_t { kPrecinctEmpty = 1 };
		
		StreamPrecinctBlockHeader					header;
		std::vector<uint8_t>						compressedData;
		PrecinctModel								arithModel;
		RLECoder<uint16_t>::BlockTable				rleBlockTable;

	public:
		size_t SizeOf() const
		{
			size_t size = 0;
			size += sizeof(header);
			size += sizeof(uint8_t) * compressedData.size();
			size += sizeof(PrecinctModel::value_type) * arithModel.size();
			size += sizeof(RLECoder<uint16_t>::BlockTable::value_type) * rleBlockTable.size();

			return size;
		}

		void Serialise(OutputStream& stream)
		{
			// Serialise the precinct header
			header.magic = MagicNumbers::kPrecintHeader;
			header.sizePrecinctModelPMFTable = arithModel.size();
			header.sizeCompressedPrecinct = compressedData.size();
			header.sizeRLEBlockTable = rleBlockTable.size();
			stream << header;

			// Serialise the coder
			for (const auto& entry : arithModel)
			{
				stream << entry.first << entry.second;
			}

			// Serialise the RLE block table
			stream << rleBlockTable;

			// Serialise the encoded data
			stream << compressedData;
		}

		void Deserialise(InputStream& stream)
		{
			// Deseriaise the precinct header
			stream >> header;
			AssertMsg(header.magic == MagicNumbers::kPrecintHeader, "Corrupt byte stream: magic number mismatch in precinct data header.");

			// Deserialise the coder
			arithModel.resize(header.sizePrecinctModelPMFTable);
			for (const auto& entry : arithModel)
			{
				stream.Read(entry.first);
				stream.Read(entry.second);
			}

			// Deserialise the RLE block table
			stream.Read(rleBlockTable, header.sizeRLEBlockTable);

			// Deserialise the encoded data
			stream.Read(compressedData, header.sizeCompressedPrecinct);
		}
	};
}