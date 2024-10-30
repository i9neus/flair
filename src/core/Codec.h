#pragma once

#include "CompressedImageData.h"
#include "Image.h"
#include "coder/ArithmeticCoder.h"
#include "wavelets/NormalisedDWT.h"

namespace HDRI
{
	enum CodecFlags : uint32_t 
	{ 
		kVerbose = 1, 
		kOutputWaveletData = 2,
		kDebugCoder = 4,
		kUseRLE = 8,
		kRemapHilbert = 16
	};
	
	class Codec
	{
	private:
		enum ChannelTypes : int { kChannelY, kChannelU, kChannelV };
		using CoderModelType = uint16_t;

	public:
		Codec(const uint32_t flags = 0u);

		void				Serialise(const std::string& outputPath, const Image3f& inputImage);
		Image3f				Deserialise(const std::string& inputPath, Image3f& outputImage);

		void      			Encode(const Image3f& inputImage, CompressedImageData& decomposed);
		void				Decode(const CompressedImageData& decomposed, Image3f& outputImage);

		void				DecodeWaveletCoeffs(const Image3f& waveletCoeffs, Image3f& outputImage) const;
		const Image3f&		GetWaveletData() const { return m_waveletCoeffs; }
	private:
		void				PrepareEncoder(const int width, const int height);
		void				PrepareDecoder(const CompressedImageData& image);

		void				EncodeChannel(Image1f& chnlData, Image1f* waveletData, const int chnlIdx, CompressedChannelData& decompData);
		void				DecodeChannel(const CompressedChannelData& decompData, const int chnlIdx, Image1f& chnlData);

		template<typename... Pack>
		inline void Log(const char* fmt, Pack... pack)
		{
			if (m_flags & kVerbose)
			{
				std::printf(fmt, pack...);
			}
		}

	private:
		Image3f				m_waveletCoeffs;
		uint32_t			m_flags;
		NormalisedDWT<CDF97<float>> m_dwt;

		int					m_width;
		int					m_height;
		int					m_area;
		int					m_numPrecincts;
		float				m_quality;

		std::vector<int>	m_quantRatesY;
		std::vector<int>	m_quantRatesUV;
		float				m_thresholdY;
		float				m_thresholdUV;
		float				m_gamma;
		int					m_minCompressedPrecinct;
		bool				m_normaliseCoeffs;
		int					m_maxQuant;
		int					m_minQuantY;
		int					m_minQuantUV;
	};
}