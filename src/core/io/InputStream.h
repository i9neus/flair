#pragma once

#include "../Includes.h"
#include <fstream>
#include "OutputStream.h"

namespace Flair
{
	class InputStream
	{
	private:
		InputStream() : m_dataSize(0), m_readHead(0), m_data(nullptr) {}

	public:
		template<typename Type>
		InputStream(const Type* rawData, const size_t bufferSize) : InputStream()
		{
			m_data = reinterpret_cast<const uint8_t*>(rawData);
			m_dataSize = bufferSize * sizeof(Type);
			m_readHead = 0;
		}

		InputStream(const OutputMemStream& other) : InputStream()
		{
			m_data = other.Data().data();
			m_dataSize = other.Data().size();
		}

		InputStream(const std::string& path) : InputStream()
		{
			// Open the file and determine its size
			std::ifstream file(path, std::ios::in | std::ios::binary);
			AssertFmt(file.is_open(), "Error: file '%s' not found", path.c_str());

			file.seekg(0, std::ios::end);
			m_dataSize = file.tellg();
			file.seekg(0, std::ios::beg);

			std::string code;
			m_fileData.reset(new uint8_t[m_dataSize]);
			file.read(reinterpret_cast<char*>(m_fileData.get()), m_dataSize);

			m_data = m_fileData.get();
		}

		void Seek(const size_t pos)
		{
			m_readHead = std::min(pos, m_dataSize - 1);
		}

		template<typename T>
		void Read(std::vector<T>& data, const size_t numElements)
		{
			if (numElements > 0)
			{
				data.resize(numElements);
				ReadData(data.data(), sizeof(T) * numElements);
			}
		}

		template<typename T> inline void Read(T& data) { ReadData((void*)&data, sizeof(T)); }
		template<typename T> inline InputStream& operator>>(T& data) { ReadData(&data, sizeof(T)); return *this; }

		const uint8_t* Data() const { return m_data; }
		size_t Size() const { return m_dataSize; }

	private:
		inline void ReadData(void* data, const size_t dataSize)
		{
			AssertFmt(m_readHead + dataSize <= m_dataSize, "Tried to read past end of byte stream (size %zi bytes)", m_dataSize);
			memcpy(data, &m_data[m_readHead], dataSize);
			m_readHead += dataSize;
		}
		std::unique_ptr<uint8_t> m_fileData;
		const uint8_t*			m_data;
		size_t					m_readHead;
		size_t					m_dataSize;
	};
}