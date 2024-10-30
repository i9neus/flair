#pragma once

#include "Includes.h"

namespace HDRI
{
	class ByteStream
	{
	public:
		ByteStream() { Clear();  }

		void Clear() 
		{ 
			m_checksum = 0xab;
			m_data.clear(); 
		}

		void Seek(const size_t pos)
		{
			m_idx = std::min(pos, m_data.size() - 1);
		}

		template<typename T>
		ByteStream& operator<<(const T& data)
		{
			static_assert(std::is_standard_layout<T>::value, "Not standard layout type.");

			WriteData(&data, sizeof(T));
			return *this;
		}

		template<typename T>
		ByteStream& operator<<(const std::vector<T>& vec)
		{
			if (!vec.empty())
			{
				WriteData(vec.data(), sizeof(T) * vec.size());
			}
			return *this;
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
		
		template<typename T> inline void Write(const std::vector<T>& vec) { this->operator<<(vec); }
		template<typename T> inline void Read(T& data) { ReadData((void*)&data, sizeof(T)); }
		template<typename T> inline ByteStream& operator>>(T& data) { ReadData(&data, sizeof(T)); return *this;  }
		
		void WriteChecksum() { m_data.push_back(m_checksum); }

		const std::vector<uint8_t>& Data() const { return m_data; }
		const size_t Size() const { return m_data.size(); }

	private:
		inline void WriteData(const void* data, const size_t dataSize)
		{
			m_data.resize(m_data.size() + dataSize);
			memcpy(&m_data[m_data.size() - dataSize], data, dataSize);
			for(auto it = m_data.end() - dataSize; it != m_data.end(); ++it)
			{
				m_checksum ^= *it; 
			}
		}

		inline void ReadData(void* data, const size_t dataSize)
		{
			AssertFmt(m_idx + dataSize <= m_data.size(), "Tried to read past end of byte stream (size %i bytes)", m_data.size());
			memcpy(data, &m_data[m_idx], dataSize);
			m_idx += dataSize;
		}

		std::vector<uint8_t>		m_data;
		size_t						m_idx;
		uint8_t						m_checksum;
	};
}