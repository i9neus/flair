#pragma once

#include "Includes.h"
#include "io/InputStream.h"
#include "io/OutputStream.h"
#include "coders/ArithmeticCoder.h"
#include "CompressedChannelData.h"

namespace Flair
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
			StreamImageHeader()
			{
				std::memset(this, 0, sizeof(StreamImageHeader));
				magic = MagicNumbers::kImageHeader;
			}
			
			MagicType										magic;
			int												width;
			int												height;
			int												channels;
			float											imageGamma;
			float											quantiseGamma;
			float											quantQuality;
			float											quantAttenuation;
			uint32_t										encoderFlags;
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

		CompressedImageData(InputStream& stream) : CompressedImageData()
		{
			Deserialise(stream);
		}

		void Serialise(OutputStream& stream)
		{
			AssertFmt(header.channels == kChannels, "Serialiser only supports 3 channels");

			// Write the file header
			using ModelType = CompressedChannelData::PrecinctModel::value_type::first_type;
			header.coderModelEntrySize = sizeof(ModelType);
			stream << header;

			for (int chnlIdx = 0; chnlIdx < kChannels; ++chnlIdx)
			{
				channelData[chnlIdx].Serialise(stream, chnlIdx);
			}
		}

		void Deserialise(InputStream& stream)
		{
			stream.Seek(0);

			stream >> header;
			AssertFmt(header.magic == MagicNumbers::kImageHeader, "Corrupt byte stream: magic number mismatch in image data header.");
			AssertFmt(header.channels == 3, "Only 3 channel images are supported.");
			AssertFmt(header.coderModelEntrySize == sizeof(CompressedChannelData::PrecinctModel::value_type::first_type), "Unexpected coder model entry size.");

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