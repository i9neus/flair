#include "Codec.h"

#include "Hilbert2D.h"
#include "wavelets/NormalisedDWT.h"
#include "Viewer/math/MathUtils.h"
#include "coder/RLECoder.h"

namespace HDRI
{
    static constexpr float kDither = 0.f;
    static constexpr int kMaxCoeff = 1;
    
    Codec::Codec(const uint32_t flags)
    {
        m_flags = kRemapHilbert | kUseRLE;
        m_flags |= flags;
        
        m_minCompressedPrecinct = 4;
        m_gamma = 2.2;
        // NOTE: Quality is a normalised parameter between -1 and 1. 
        m_quality = 0.3f;

        AssertMsg(m_minCompressedPrecinct >= 1, "minCompressedPrecinct must be >= 1");
    }

    void Codec::PrepareEncoder(const int width, const int height)
    {
        m_width = width;
        m_height = height;
        m_area = m_width * m_height;
        
        AssertMsg(m_width == m_height, "Codec only supports square images.");
        int bitsSet = 0;
        for (int i = 0; i < 31; ++i) { bitsSet += ((m_width & (1 << i)) != 0); }
        AssertFmt(bitsSet == 1, "Codec only supports image dimensions in powers of two (input is %i x %i) %i", m_width, m_height, bitsSet);
       
        m_numPrecincts = 1 + std::log2(1 + m_width);
        m_quantRatesY.resize(m_numPrecincts, m_maxQuant);
        m_quantRatesUV.resize(m_numPrecincts, m_maxQuant);
        m_thresholdY = 0.1f;

        // Maximum quantisation rate is half the size of the coder model datatype
        const float quantQuality = std::pow(saturate(m_quality), 4.0f);
        m_maxQuant = std::numeric_limits<CoderModelType>::max() / 2;
        m_minQuantY = int(std::round(mix(4., 1024., quantQuality)));
        m_minQuantUV = std::max(4, m_minQuantY / 4);
        m_thresholdY = 2 / m_minQuantY;
        m_thresholdUV = 2. / m_minQuantUV;

        std::printf("Quality %f: m_minQuantY = %i\n", m_quality, m_minQuantY);

        Log("Quantisation rates:");
        float qY = m_minQuantY, qUV = m_minQuantUV;
        const float dqY = mix(1.f, 4.f, saturate(1. + m_quality));
        const float dqUV = mix(1.f, 2.f, saturate(1. + m_quality));

        for (int i = m_quantRatesY.size() - 1; i >= 0; --i)
        {
            m_quantRatesY[i] = qY;
            m_quantRatesUV[i] = qUV;
            qY = std::min(float(m_maxQuant), qY * dqY);
            qUV = std::min(float(m_maxQuant), qUV * dqUV);

            Log("  - %i: Y: %i, UV: %i\n", i, m_quantRatesY[i], m_quantRatesUV[i]);
        }

        // Prepare the discrete wavelet transform
        m_dwt.Prepare(m_width, 0);// kDWTLogSpace);
    }

    void Codec::PrepareDecoder(const CompressedImageData& image)
    {
        m_width = image.header.width;
        m_height = image.header.height;
        m_area = m_width * m_height;
        m_quality = image.header.quality;
        m_flags = image.header.encoderFlags;
        m_numPrecincts = image.header.numPrecincts;
        m_gamma = image.header.gamma;
        m_quantRatesY = image.channelData[0].quantRates;
        m_quantRatesUV = image.channelData[1].quantRates;
        m_thresholdY = image.header.thresholdY;
        m_thresholdUV = image.header.thresholdUV;

        AssertMsg(m_width == m_height, "Codec only supports square images.");
        int bitsSet = 0;
        for (int i = 0; i < 31; ++i) { bitsSet += ((m_width & (1 << i)) != 0); }
        AssertFmt(bitsSet == 1, "Codec only supports image dimensions in powers of two (input is %i x %i) %i", m_width, m_height, bitsSet);

        // Prepare the discrete wavelet transform
        m_dwt.Prepare(m_width, 0);// kDWTLogSpace);
    }

    // Encodes a single image channel into a compressed bitstream stored in decompData
    void Codec::EncodeChannel(Image1f& chnlData, Image1f* waveletData, const int chnlIdx, CompressedChannelData& decompData)
    {
        Log("Encoding channel %i...\n", chnlIdx);
        
        // In-place transform the image using the DCT
        m_dwt.Forward(chnlData.Vector(), decompData.dwtPassNorms);

        // Threshold the resulting coefficients based on the channel type
        const float threshold = (chnlIdx == kChannelY) ? m_thresholdY : m_thresholdUV;
        //const float threshold = 0.;

        // Store the uncompressed precinct data
        decompData.header.minCompressedPrecinct = m_minCompressedPrecinct;
        decompData.header.numPrecincts = m_numPrecincts;
        decompData.uncompressedPrecinctData.resize(1);
        decompData.compressedPrecinctData.resize(m_numPrecincts);
        decompData.quantRates = (chnlIdx == 0) ? m_quantRatesY : m_quantRatesUV;

        std::vector<CoderModelType> quantisedData;

        // DC is always stored explicitly
        decompData.uncompressedPrecinctData[0] = chnlData[0];
        if (waveletData) { (*waveletData)[0] = chnlData[0]; }

        // Go precinct by precint...
        for (int quadrantSize = 1, precinctIdx = 1; 
             quadrantSize != m_width; 
             quadrantSize <<= 1, ++precinctIdx)
        {
            // Select quantisation rate based on the precinct and the index of the channel
            const CoderModelType rate = (chnlIdx == kChannelY) ? m_quantRatesY[precinctIdx] : m_quantRatesUV[precinctIdx];
            const float precinctThreshold = threshold * std::pow(0.25, float(precinctIdx));
            const int quadrantArea = sqr(quadrantSize);

            // Traverse the three quadrants of the image making up the precinct
            bool isPrecinctEmpty = true;
            quantisedData.resize(quadrantArea * 3);
            for (int v = 0, quadIdx = 0, quantIdx = 0; v < 2; ++v)
            {
                for (int u = 0; u < 2; ++u)
                {
                    if (u != 0 || v != 0)
                    {
                        for (int y = 0; y < quadrantSize; ++y)
                        {
                            int pixelIdx = (v * quadrantSize + y) * m_width + (u * quadrantSize);
                            for (int x = 0; x < quadrantSize; ++x, ++pixelIdx, ++quantIdx)
                            {
                                float f = chnlData[pixelIdx];

                                // For the lowest precincts, we don't compress the data but instead store it explicitly
                                if (precinctIdx < m_minCompressedPrecinct)
                                {
                                    decompData.uncompressedPrecinctData.push_back(f);
                                }
                                else
                                {
                                    // Apply deadzone (thresholding) by offsetting and clamping
                                    f = std::max(0.f, std::abs(f) - precinctThreshold) * sign(f);

                                    // Scale to the quantisation rate and apply dithering
                                    f = std::max(0.f, kDither + std::abs(f / kMaxCoeff) * rate) * sign(f);

                                    // Quantise to integer
                                    const CoderModelType quant = clamp(int(f) + (rate - 1), 0, 2 * rate);

                                    // Store the coefficient, remapping to a Hilbert curve if necessary
                                    const auto qIdx = (m_flags & kRemapHilbert) ? (quadIdx * quadrantArea + Hilbert2D::ToCurve(quadrantSize, x, y)) : quantIdx;
                                    quantisedData[qIdx] = quant;

                                    // Check for constant data
                                    if (quant != rate - 1) { isPrecinctEmpty = false; }

                                    if (waveletData)
                                    {
                                        // Store the wavelet coefficients for diagnostics
                                        (*waveletData)[pixelIdx] = kMaxCoeff * (float(quant) - (rate - 1)) / rate;
                                        //(*waveletData)[pixelIdx] = std::abs(chnlData[pixelIdx]);
                                    }
                                }
                            }
                        }
                        ++quadIdx;
                    }
                }
            }

            if (precinctIdx >= m_minCompressedPrecinct)
            {
                auto& compressedPrecinct = decompData.compressedPrecinctData[precinctIdx];
                
                // If the precinct contains no data (common with highly compressed channels), we don't need to compress anything
                if (isPrecinctEmpty)
                {
                    compressedPrecinct.header.flags |= CompressedChannelData::kPrecinctEmpty;
                }
                else
                {
                    ArithmeticCoder<CoderModelType> arithCoder;
                    if (m_flags & kUseRLE)
                    {
                        // Run-length encode the quantised coefficients
                        std::vector<CoderModelType> rleData;
                        RLECoder<CoderModelType>::Encode(quantisedData, rate - 1, 32, rleData, compressedPrecinct.rleBlockTable);

                        // Arithmetic encode the RLE data
                        arithCoder.Build(rleData, (m_flags & kDebugCoder) != 0);
                        compressedPrecinct.compressedData = arithCoder.Encode(rleData); Assert(!compressedPrecinct.compressedData.empty());
                        compressedPrecinct.arithModel = arithCoder.GetModel(); Assert(!compressedPrecinct.arithModel.empty());
                        Log("  - %i : %i -> %i bytes\n", precinctIdx, sizeof(CoderModelType) * rleData.size(), sizeof(uint8_t) * compressedPrecinct.compressedData.size() + compressedPrecinct.rleBlockTable.size() * sizeof(uint16_t));
                        Log("         Table size: %i\n", compressedPrecinct.rleBlockTable.size() * sizeof(uint16_t));
                    }
                    else
                    {
                        // Encode the quantised coefficients and store the bitstream along with the derived coding model
                        arithCoder.Build(quantisedData, (m_flags & kDebugCoder) != 0);
                        compressedPrecinct.compressedData = arithCoder.Encode(quantisedData);
                        compressedPrecinct.arithModel = arithCoder.GetModel(); Assert(!compressedPrecinct.arithModel.empty());
                        Log("  - %i : %i -> %i bytes\n", precinctIdx, sizeof(CoderModelType) * quantisedData.size(), sizeof(uint8_t) * compressedPrecinct.compressedData.size());
                    }
                }
            }
        }
    }

    void Codec::DecodeChannel(const CompressedChannelData& decompData, const int chnlIdx, Image1f& chnlData)
    {
        ArithmeticCoder<CoderModelType> arithCoder;
        chnlData.Erase();
        
        // DC is always stored explicitly
        auto explicitIt = decompData.uncompressedPrecinctData.begin();
        chnlData[0] = *explicitIt;
        ++explicitIt;

        const float threshold = (chnlIdx == kChannelY) ? m_thresholdY : m_thresholdUV;
        
        // Go precinct by precint...
        for (int quadrantSize = 1, precinctIdx = 1;
            quadrantSize != m_width; 
            quadrantSize <<= 1, ++precinctIdx)
        {       
            auto& compressedPrecinct = decompData.compressedPrecinctData[precinctIdx];

            // If this precinct contains no data, skip it and move on
            if (compressedPrecinct.header.flags & CompressedChannelData::kPrecinctEmpty) { continue; }
            
            // Select quantisation rate based on the index of the channel
            const CoderModelType rate = (chnlIdx == kChannelY) ? m_quantRatesY[precinctIdx] : m_quantRatesUV[precinctIdx];
            const float precinctThreshold = threshold * std::pow(0.25, float(precinctIdx));
            const int quadrantArea = sqr(quadrantSize);
            
            std::vector<CoderModelType> quantisedData;
            if (precinctIdx >= decompData.header.minCompressedPrecinct)
            {
                // Decode the bitstream using the associated model
                arithCoder.SetModel(compressedPrecinct.arithModel);

                if (m_flags & kUseRLE)
                {
                    Assert(!compressedPrecinct.rleBlockTable.empty());
                    std::vector<CoderModelType> rleData = arithCoder.Decode(compressedPrecinct.compressedData);                    
                    quantisedData = RLECoder<uint16_t>::Decode(rleData, rate - 1, compressedPrecinct.rleBlockTable);                  
                }
                else
                {
                    quantisedData = arithCoder.Decode(compressedPrecinct.compressedData);
                }    

                // Sanity check
                AssertFmt(quantisedData.size() == quadrantArea * 3,
                    "Size of decoded precinct data does not match the area of the precinct (is %i, should be %i)", quantisedData.size(), quadrantArea * 3);
            }

            // Dequantise the precinct
            for (int v = 0, quadIdx = 0, quantIdx = 0; v < 2; ++v)
            {
                for (int u = 0; u < 2; ++u)
                {
                    if (u != 0 || v != 0)
                    {
                        for (int y = 0; y < quadrantSize; ++y)
                        {
                            int pixelIdx = (v * quadrantSize + y) * m_width + (u * quadrantSize);
                            for (int x = 0; x < quadrantSize; ++x, ++pixelIdx, ++quantIdx)
                            {
                                float& f = chnlData[pixelIdx];

                                if (precinctIdx < decompData.header.minCompressedPrecinct)
                                {
                                    Assert(explicitIt != decompData.uncompressedPrecinctData.end());
                                    f = *explicitIt;
                                    ++explicitIt;
                                }
                                else
                                {
                                    // Dequantise
                                    const auto qIdx = (m_flags & kRemapHilbert) ? (quadIdx * quadrantArea + Hilbert2D::ToCurve(quadrantSize, x, y)) : quantIdx;
                                    int quant = quantisedData[qIdx] - (rate - 1);
                                    f = float(kMaxCoeff * quant) / rate;

                                    // Compensate for the dead zone. If value is non-zero, apply the threshold based on the sign of the coefficient
                                    if (quant != 0) { f += precinctThreshold * sign(f); }
                                }
                            }
                        }

                        ++quadIdx;
                    }
                }
            }
        }

        // In-place inverse transform the image using the DCT
        m_dwt.Inverse(chnlData.Vector(), decompData.dwtPassNorms);
    }

    void Codec::Encode(const Image3f& inputImage, CompressedImageData& compImage)
    {
        PrepareEncoder(inputImage.Width(), inputImage.Height());

        compImage.header.width = inputImage.Width();
        compImage.header.height = inputImage.Height();
        compImage.header.gamma = m_gamma;
        compImage.header.numPrecincts = m_numPrecincts;
        compImage.header.thresholdY = m_thresholdY;
        compImage.header.thresholdUV = m_thresholdUV;
        compImage.header.quality = m_quality;
        compImage.header.encoderFlags = m_flags;
        
        Image3f remappedImage = inputImage;

        // Transform the image to gamma-corrected space
        remappedImage.ApplyGamma(1. / m_gamma);

        // Transform into YUV colour space
        remappedImage.RGBToYUV();

        // If we're outputting diagnostic wavelet coefficients, initialise the buffer here
        Image1f chnlData(inputImage.Width(), inputImage.Height());
        std::unique_ptr<Image1f> waveletChnlData;
        if (m_flags & kOutputWaveletData)
        {
            m_waveletCoeffs.Resize(inputImage.Width(), inputImage.Height());
            waveletChnlData.reset(new Image1f(inputImage.Width(), inputImage.Height()));
        }
        
        // Encode the image channel by channel
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            remappedImage.ExtractChannel(chnlData, chnlIdx);

            EncodeChannel(chnlData, waveletChnlData.get(), chnlIdx, compImage.channelData[chnlIdx]);

            if (waveletChnlData)
            {
                m_waveletCoeffs.EmplaceChannel(*waveletChnlData, chnlIdx);
            }
        }
    }

    /*void Codec::DecodeWaveletCoeffs(const Image3f& waveletCoeffs, Image3f& outputImage) const
    {
        outputImage.Resize(waveletCoeffs);
        
        // Encode the image channel by channel
        Image1f chnlData(waveletCoeffs.Width(), waveletCoeffs.Height());
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            waveletCoeffs.ExtractChannel(chnlData, chnlIdx);

            // In-place inverse transform the image using the DCT
            DWT<CDF97<float>>::Inverse(chnlData.Vector(), m_width, m_normaliseCoeffs);
            
            outputImage.EmplaceChannel(chnlData, chnlIdx);
        }

        // Transform back into RGB colour space
        outputImage.YUVToRGB();

        // Transform the image to gamma-corrected space
        outputImage.ApplyGamma(m_gamma);
    }*/

    void Codec::Decode(const CompressedImageData& compImage, Image3f& outputImage)
    {
        PrepareDecoder(compImage);
        //PrepareEncoder(compImage.width, compImage.height);
        
        /*m_gamma = compImage.gamma;
        m_numPrecincts = compImage.numPrecincts;
        m_quantRatesY = compImage.channelData[0].quantRates;
        m_quantRatesUV = compImage.channelData[1].quantRates;*/

        outputImage.Resize(m_width, m_height);
        Image1f chnlData(m_width, m_height);

        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            DecodeChannel(compImage.channelData[chnlIdx], chnlIdx, chnlData);

            outputImage.EmplaceChannel(chnlData, chnlIdx);
        }

        // Transform back into RGB colour space
        outputImage.YUVToRGB();

        // Transform the image to gamma-corrected space
        outputImage.ApplyGamma(m_gamma);
    }
}
