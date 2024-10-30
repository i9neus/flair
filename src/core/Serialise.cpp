#include "Serialise.h"

namespace HDRI
{
	void Serialise(const DecomposedImage& decompData, ByteStream& stream)
	{
		decompData.Serialise(stream);
	}

	void Deserialise(const ByteStream& bitStream, DecomposedImage& decompData)
	{

	}
}