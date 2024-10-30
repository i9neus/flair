#pragma once

#include "Includes.h"
#include "ByteStream.h"
#include "coder/ArithmeticCoder.h"
#include "CompressedChannelData.h"

namespace HDRI
{
	namespace MagicNumbers
	{
		static constexpr MagicType kImageHeader = 0xba45f007;
	};
	
	struct CompressedImageData
	{
	private:
		static constexpr int kChannels = 3;

		// Stored at the beginning of the stream
		struct StreamImageHeader
		{
			MagicType										magic = MagicNumbers::kImageHeader;
			int												width;
			int												height;
			int												channels;
			float											gamma;
			float											quality;
			uint32_t										encoderFlags;
			float											thresholdY;
			float											thresholdUV;
			int												numPrecincts;

			int												coderModelEntrySize;
		};

	public:
		StreamImageHeader								header;
		std::array<CompressedChannelData, 3>			channelData;

	public:
		CompressedImageData()
		{
			header.channels = 3;
		}

		void Serialise(ByteStream& stream)
		{
			AssertMsg(header.channels == kChannels, "Serialiser only supports 3 channels");

			stream.Clear();

			// Write the file header
			using ModelType = CompressedChannelData::PrecinctModel::value_type::first_type;
			header.coderModelEntrySize = sizeof(ModelType);
			stream << header;

			for (int chnlIdx = 0; chnlIdx < kChannels; ++chnlIdx)
			{
				channelData[chnlIdx].Serialise(stream, chnlIdx);
			}
		}

		void Deserialise(ByteStream& stream)
		{
			stream.Seek(0);

			stream >> header;
			AssertMsg(header.magic == MagicNumbers::kImageHeader, "Corrupt byte stream: magic number mismatch in image data header.");
			AssertMsg(header.channels == 3, "Only 3 channel images are supported.");
			AssertMsg(header.coderModelEntrySize == sizeof(CompressedChannelData::PrecinctModel::value_type::first_type), "Unexpected coder model entry size.");

			for (int chnlIdx = 0; chnlIdx < kChannels; ++chnlIdx)
			{
				channelData[chnlIdx].Deserialise(stream, chnlIdx);
			}
		}

		size_t SizeOf() const
		{
			size_t size = sizeof(header.width) + sizeof(header.height);
			for (const auto& data : channelData) { size += data.SizeOf(); }

			return size;
		}
	};
}