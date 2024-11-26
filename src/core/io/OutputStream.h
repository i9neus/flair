#pragma once

#include "../Includes.h"
#include <fstream>

namespace Flair
{
	class OutputStream
	{
	public:
		OutputStream() = default;

		template<typename T>
		OutputStream& operator<<(const T& data)
		{
			static_assert(std::is_standard_layout<T>::value, "Not standard layout type.");

			WriteData(&data, sizeof(T));
			return *this;
		}

		template<typename T>
		OutputStream& operator<<(const std::vector<T>& vec)
		{
			if (!vec.empty())
			{
				WriteData(vec.data(), sizeof(T) * vec.size());
				//UpdateChecksum();
			}
			return *this;
		}
		
		template<typename T> inline void Write(const std::vector<T>& vec) { this->operator<<(vec); }		

		virtual size_t Size() const = 0;
		virtual void Flush() {}

	private:
		virtual void WriteData(const void* data, const size_t dataSize) = 0;
		
		void UpdateChecksum(const void* data, const size_t dataSize)
		{
			auto ptr = reinterpret_cast<const char*>(data);
			for (int i = 0; i < dataSize; ++i)
			{
				m_checksum ^= ptr[i];
			}
		}

		uint8_t						m_checksum;
	};

	// Writes serialised data to a buffer
	class OutputMemStream : public OutputStream
	{
	public:
		const std::vector<uint8_t>& Data() const { return m_data; }
		virtual size_t Size() const override final { return m_data.size(); }

	protected:
		virtual void WriteData(const void* data, const size_t dataSize) override final
		{
			m_data.resize(m_data.size() + dataSize);
			memcpy(&m_data[m_data.size() - dataSize], data, dataSize);
		}

	private:
		std::vector<uint8_t>		m_data;
		uint8_t						m_checksum;
	};

	// Writes serialised data directly to file
	class OutputFileStream : public OutputStream
	{
	public:
		OutputFileStream(const std::string& path) : 
			OutputStream(),
			m_bytesWritten(0)
		{
			m_file.open(path, std::ios::out | std::ios::binary);
			AssertFmt(m_file.is_open(), "Error: could not open '%s' for writing.", path.c_str());
		}

		~OutputFileStream() { Close(); }

		void Close()
		{
			m_file.flush();
			m_file.close();
		}

		virtual size_t Size() const override final { return m_bytesWritten; }
		virtual void Flush() override final { m_file.flush(); }

	protected:
		virtual void WriteData(const void* data, const size_t dataSize) override final
		{
			Assert(m_file.is_open());
			m_file.write(reinterpret_cast<const char*>(data), dataSize);
			m_bytesWritten += dataSize;
		}

	private:
		std::ofstream m_file;
		size_t m_bytesWritten;
	};
}