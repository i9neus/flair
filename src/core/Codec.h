#pragma once

#include "CompressedImageData.h"
#include "image/Image.h"
#include "coders/ArithmeticCoder.h"
#include "math/wavelets/1d/NormalisedDWT1.h"

namespace Flair
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
	public:
		struct Params
		{
			Params();
			Params& operator=(const Params& other);

			uint32_t			flags;
			float				quantQuality;
			float				quantAttenuation;
			float				imageGamma;
			float			    quantiseGamma;
			float				threshold;
			float				thresholdAttenuation;
			int					minCompressedPrecinct;
			int					minQuant;
			float				quantRounding;
			int					dwtFlags;
		};

	private:
		enum ChannelTypes : int { kChannelY, kChannelU, kChannelV };
		using CoderModelType = uint16_t;

	public:
		Codec(const uint32_t flags = 0u);

		void      			Encode(const Image3f& inputImage, CompressedImageData& decomposed);
		void				Decode(const CompressedImageData& decomposed, Image3f& outputImage);

		void				EncodeWaveletCoeffs(const Image3f& inputImage, Image3f& waveletCoeffs) const;
		void				DecodeWaveletCoeffs(const Image3f& waveletCoeffs, Image3f& outputImage) const;
		const Image3f&		GetWaveletData() const { return m_waveletCoeffs; }

		const Params&       GetEncoderParams() const { return m_params; }
		void				SetEncoderParams(const Params& params);

	private:
		void				PrepareEncoder(const int width, const int height);
		void				PrepareDecoder(const CompressedImageData& image);

		void				EncodeChannel(Image1f& chnlData, Image1f* waveletData, const int chnlIdx, CompressedChannelData& decompData);
		Image1f				DecodeChannel(const CompressedChannelData& decompData, const int chnlIdx);

		void				ForwardDWT(Image1f& chnlData) const;
		void				InverseDWT(Image1f& chnlData) const;


		inline float		GetPrecinctThreshold(const int precinctIdx, const CoderModelType rate) const;

		template<typename... Pack>
		inline void Log(const char* fmt, Pack... pack)
		{
			if (m_params.flags & kVerbose)
			{
				std::cout << tfm::format(fmt, pack...);
			}
		}

	private:
		Params				m_params;

		int					m_width;
		int					m_height;
		int					m_area;
		int					m_numPrecincts;

		std::vector<int>	m_quantRatesY;
		std::vector<int>	m_quantRatesUV;
		bool				m_normaliseCoeffs;
		int					m_maxQuant;
		int					m_minQuantY;
		int					m_minQuantUV;

		Image3f				m_waveletCoeffs;
	};
}