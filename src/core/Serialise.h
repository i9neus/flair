#pragma once

#include "DecomposedImage.h"
#include "ByteStream.h"

namespace HDRI
{	
	void Serialise(const DecomposedImage& decompData, ByteStream& stream);
	void Deserialise(ByteStream& stream, DecomposedImage& decompData);
}