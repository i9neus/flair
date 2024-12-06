#include "Codec.h"

#include "math/Hilbert2D.h"
#include "math/Hash.h"
#include "math/wavelets/1d/NormalisedDWT1.h"
#include "math/wavelets/2d/StaticDWT2.h"
#include "core/math/MathUtils.h"
#include "coders/RLECoder.h"
#include "image/ImageOps.h"

#ifdef FLAIR_ENABLE_MULTITHREADING
#include <thread>
#endif

namespace Flair
{
    // These parameters determine the performance of the compressor. Only the quality setting should be exposed to the user of the library.
    Codec::Params::Params()
    {
        // Configuation flags for the primary codec.
        // kUseRLE            = Apply run-length encoding to the quantised coefficients
        // kRemapHilbert      = Remap quantised coefficients to a hilbert curve to make RLE coding more effective
        flags = kRemapHilbert | kUseRLE;        

        // Quality is a normalised parameter between 0 and 1. It determines the base quantisation rate i.e. the aggressiveness of the quantisation rate at the finest precinct.
        quantQuality = 0.3;

        // Attenuation determine how lower precincts are more highly quantised than higher ones. The default is 4.
        quantAttenuation = 2.;
        
        // The lowest precinct below which data aren't compressed but instead are written out as half-precision floats
        minCompressedPrecinct = 4;
        
        // The gamma space of the input image
        imageGamma = 2.2;

        // The gamma space used to quantise coefficient. Higher value = greater capacity to distinguish fine details.
        quantiseGamma = 1.;

        // The amount of thresholding to be applied as a proportion of the per-precinct quantisataion rate
        threshold = 0.5;

        // How much the coefficient thresholding is attenuated based on the precinct index (where precinctThreshold = threshold * thresholdAttenuation ^ (numPrecincts - precinctIdx - 1))
        thresholdAttenuation = 1.;

        // The minimum quantisation level 
        minQuant = 4;

        // How quantised coefficients are rounded. A value of 0.5 is the same as calling std::round(q)
        quantRounding = 0.;

        // The flags used to initialise the wavelet transforms
        dwtFlags = kDWTLogSpace;
    }

    Codec::Params& Codec::Params::operator=(const Codec::Params& other)
    {
        std::memcpy(this, &other, sizeof(Codec::Params));

        quantQuality = clamp(quantQuality, -1.f, 1.f);
        minCompressedPrecinct = std::max(0, minCompressedPrecinct);
        imageGamma = clamp(imageGamma, 1e-3f, 100.f);
        threshold = std::max(0.f, threshold);
        thresholdAttenuation = clamp(thresholdAttenuation, 1.f, 1e-3f);

        return *this;
    }
    
    Codec::Codec(const uint32_t flags)
    {
        m_params.flags |= flags;     

        AssertMsg(m_params.minCompressedPrecinct >= 1, "minCompressedPrecinct must be >= 1");
    }

    void Codec::SetEncoderParams(const Params& params)
    {
        m_params = params;
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

        // Maximum quantisation rate is half the size of the coder model datatype
        const float quantQuality = std::pow(saturate(m_params.quantQuality), 4.0f);
        m_maxQuant = std::numeric_limits<CoderModelType>::max() / 2;
        m_minQuantY = int(std::round(mix(float(m_params.minQuant), 1024.f, quantQuality)));
        m_minQuantUV = std::max(m_params.minQuant, m_minQuantY / 4);

        // For negative quality values, the quanisation rates are increased more slowly so that lower precincts are quantised more aggressively
        Log("Quantisation rates:\n");
        float qY = m_minQuantY, qUV = m_minQuantUV;
        for (int i = m_quantRatesY.size() - 1; i >= 0; --i)
        {
            m_quantRatesY[i] = qY;
            m_quantRatesUV[i] = qUV;
            qY = std::min(float(m_maxQuant), qY * m_params.quantAttenuation);
            qUV = std::min(float(m_maxQuant), qUV * m_params.quantAttenuation);

            Log("  - %i: Y: %i, UV: %i\n", i, m_quantRatesY[i], m_quantRatesUV[i]);
        }
    }

    void Codec::PrepareDecoder(const CompressedImageData& image)
    {        
        m_width = image.header.width;
        m_height = image.header.height;
        m_area = m_width * m_height;
        m_numPrecincts = image.header.numPrecincts;
        m_quantRatesY = image.channelData[0].quantRates;
        m_quantRatesUV = image.channelData[1].quantRates;
        m_params.quantQuality = image.header.quantQuality;
        m_params.quantAttenuation = image.header.quantAttenuation;
        m_params.imageGamma = image.header.imageGamma;
        m_params.quantiseGamma = image.header.quantiseGamma;
        m_params.flags = image.header.encoderFlags;
        

        AssertMsg(m_width == m_height, "Codec only supports square images.");
        int bitsSet = 0;
        for (int i = 0; i < 31; ++i) { bitsSet += ((m_width & (1 << i)) != 0); }
        AssertFmt(bitsSet == 1, "Codec only supports image dimensions in powers of two (input is %i x %i) %i", m_width, m_height, bitsSet);
    }

    // Encodes a single image channel into a compressed bitstream stored in decompData
    void Codec::EncodeChannel(Image1f& chnlData, Image1f* waveletData, const int chnlIdx, CompressedChannelData& decompData)
    {
        // Decompose the channel data using the discrete wavelet transform
        //NormalisedDWT1<CDF97<float>> dwt(m_width, m_params.dwtFlags);
        //dwt.Forward(chnlData.Vector(), decompData.dwtPassNorms);
        StaticDWT2<float> dwt(m_width);
        dwt.Forward(chnlData.Vector());

        // Store the uncompressed precinct data
        decompData.header.minCompressedPrecinct = m_params.minCompressedPrecinct;
        decompData.header.numPrecincts = m_numPrecincts;
        decompData.uncompressedPrecinctData.resize(1);
        decompData.compressedPrecinctData.resize(m_numPrecincts);
        decompData.quantRates = (chnlIdx == 0) ? m_quantRatesY : m_quantRatesUV;

        std::vector<CoderModelType> quantisedData;

        // DC is always stored explicitly
        decompData.uncompressedPrecinctData[0] = chnlData[0];
        if (waveletData) { (*waveletData)[0] = std::abs(chnlData[0]); }

        // Go precinct by precint...
        for (int quadrantSize = 1, precinctIdx = 1; 
             quadrantSize != m_width; 
             quadrantSize <<= 1, ++precinctIdx)
        {
            // Select quantisation rate based on the precinct and the index of the channel
            const CoderModelType rate = (chnlIdx == kChannelY) ? m_quantRatesY[precinctIdx] : m_quantRatesUV[precinctIdx];
            const float precinctThreshold = GetPrecinctThreshold(precinctIdx, rate);
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
                                if (precinctIdx < m_params.minCompressedPrecinct)
                                {
                                    decompData.uncompressedPrecinctData.push_back(f);
                                }
                                else
                                {
                                    const int32_t signf = sign(f);

                                    // Map into the quantised gamma space
                                    f = std::pow(std::abs(f), 1 / m_params.quantiseGamma);
                                    
                                    // Apply deadzone (thresholding) by offsetting and clamping
                                    f = std::max(0.f, f - precinctThreshold) / (1. - precinctThreshold);                                 

                                    // Quantise to integer               
                                    int32_t quant = int(m_params.quantRounding + f * rate) * signf + rate;

                                    // Store the coefficient, remapping to a Hilbert curve if necessary
                                    const auto qIdx = (m_params.flags & kRemapHilbert) ? (quadIdx * quadrantArea + Hilbert2D::ToCurve(quadrantSize, x, y)) : quantIdx;
                                    quantisedData[qIdx] = CoderModelType(clamp(quant, 0, int32_t(std::numeric_limits<CoderModelType>::max())));

                                    // Check for constant data
                                    if (f != 0) { isPrecinctEmpty = false; }

                                    if (waveletData)
                                    {
                                        // Store the wavelet coefficients for diagnostics
                                        auto& g = (*waveletData)[pixelIdx];
                                        g = (float(quant) - rate) / rate;
                                        g = std::abs(g);
                                        g = std::abs(chnlData[pixelIdx]);//     <--- Output unquantised wavelet coefficients
                                        g = std::pow(g, 2.2f);
                                    }
                                }
                            }
                        }
                        ++quadIdx;
                    }
                }
            }

            if (precinctIdx >= m_params.minCompressedPrecinct)
            {
                auto& compressedPrecinct = decompData.compressedPrecinctData[precinctIdx];

                // If the precinct contains no data (common with highly compressed channels), we don't need to compress anything
                if (isPrecinctEmpty)
                {
                    compressedPrecinct.header.flags |= CompressedPrecinctData::kPrecinctEmpty;
                }
                else
                {
                    ArithmeticCoder<CoderModelType> arithCoder;
                    if (m_params.flags & kUseRLE)
                    {
                        // Run-length encode the quantised coefficients
                        std::vector<CoderModelType> rleData;
                        RLECoder<CoderModelType>::Encode(quantisedData, rate - 1, 32, rleData, compressedPrecinct.rleBlockTable);

                        // Arithmetic encode the RLE data
                        arithCoder.Build(rleData, (m_params.flags & kDebugCoder) != 0);
                        compressedPrecinct.compressedData = arithCoder.Encode(rleData);
                        compressedPrecinct.arithModel = arithCoder.GetModel();
                        Assert(!compressedPrecinct.compressedData.empty());
                        Assert(!compressedPrecinct.arithModel.empty());
                    }
                    else
                    {
                        // Encode the quantised coefficients and store the bitstream along with the derived coding model
                        arithCoder.Build(quantisedData, (m_params.flags & kDebugCoder) != 0);
                        compressedPrecinct.compressedData = arithCoder.Encode(quantisedData);
                        compressedPrecinct.arithModel = arithCoder.GetModel(); Assert(!compressedPrecinct.arithModel.empty());
                    }
                }
            }
        }
    }

    float Codec::GetPrecinctThreshold(const int precinctIdx, const CoderModelType rate) const
    {
        return (m_params.threshold / rate) * std::pow(m_params.thresholdAttenuation, float(m_numPrecincts - precinctIdx - 1));
    }

    Image1f Codec::DecodeChannel(const CompressedChannelData& decompData, const int chnlIdx)
    {
        ArithmeticCoder<CoderModelType> arithCoder;
        Image1f chnlData(m_width, m_height);

        // DC is always stored explicitly
        auto explicitIt = decompData.uncompressedPrecinctData.begin();
        chnlData[0] = *explicitIt;
        ++explicitIt;

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
            const float precinctThreshold = GetPrecinctThreshold(precinctIdx, rate);
            const int quadrantArea = sqr(quadrantSize);
            
            std::vector<CoderModelType> quantisedData;
            if (precinctIdx >= decompData.header.minCompressedPrecinct)
            {
                // Decode the bitstream using the associated model
                arithCoder.SetModel(compressedPrecinct.arithModel);

                if (m_params.flags & kUseRLE)
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
                    "Size of decoded precinct data does not match the area of the precinct (is %zi, should be %i)", quantisedData.size(), quadrantArea * 3);
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

                                // For uncompressed precincts, just copy the data straight over
                                if (precinctIdx < decompData.header.minCompressedPrecinct)
                                {
                                    Assert(explicitIt != decompData.uncompressedPrecinctData.end());
                                    f = *explicitIt;
                                    ++explicitIt;
                                }
                                else
                                {
                                    // Dequantise
                                    const auto qIdx = (m_params.flags & kRemapHilbert) ? (quadIdx * quadrantArea + Hilbert2D::ToCurve(quadrantSize, x, y)) : quantIdx;
                                    int quant = quantisedData[qIdx] - rate; 
                                    f = float(quant);
                                    f = sign(f) * std::max(0.f, std::abs(f) - m_params.quantRounding) / rate;

                                    // Compensate for the dead zone. If value is non-zero, apply the threshold based on the sign of the coefficient
                                    if (quant != 0) 
                                    { 
                                        f = mix(precinctThreshold, 1.f, std::abs(f)) * sign(f);
                                    }

                                    f = std::pow(std::abs(f), m_params.quantiseGamma) * sign(f);
                                }
                            }
                        }

                        ++quadIdx;
                    }
                }
            }
        }

        // Invert the wavelet transform of the decoded coefficients
        //NormalisedDWT1<CDF97<float>> dwt(m_width, m_params.dwtFlags);
        //dwt.Inverse(chnlData.Vector(), decompData.dwtPassNorms);
        StaticDWT2<float> dwt(m_width);
        dwt.Inverse(chnlData.Vector());

        return chnlData;
    }

    void Codec::Encode(const Image3f& inputImage, CompressedImageData& compImage)
    {
        compImage = CompressedImageData();
        
        PrepareEncoder(inputImage.Width(), inputImage.Height());

        compImage.header.width = inputImage.Width();
        compImage.header.height = inputImage.Height();
        compImage.header.imageGamma = m_params.imageGamma;
        compImage.header.quantiseGamma = m_params.quantiseGamma;
        compImage.header.numPrecincts = m_numPrecincts;
        compImage.header.quantQuality = m_params.quantQuality;
        compImage.header.quantAttenuation = m_params.quantAttenuation;
        compImage.header.encoderFlags = m_params.flags;
        
        Image3f remappedImage = inputImage;

        // Transform the image to gamma-corrected space
        remappedImage.ApplyGamma(1. / m_params.imageGamma);

        // Transform into YUV colour space
        remappedImage.RGBToYUV();

        // If we're outputting diagnostic wavelet coefficients, initialise the buffer here
        if (m_params.flags & kOutputWaveletData)
        {
            m_waveletCoeffs.Resize(inputImage.Width(), inputImage.Height());
        }

        // Functor that extracts a channel from the image, encodes it, then emplaces any generated wavelet coefficients
        std::function<void(int)> EncodeChannelFunctor = [&, this](int chnlIdx)
        {
            std::unique_ptr<Image1f> waveletChnlData;
            if (m_params.flags & kOutputWaveletData)
            {
                waveletChnlData.reset(new Image1f(inputImage.Width(), inputImage.Height()));
            }
            
            Image1f chnlData = remappedImage.ExtractChannel(chnlIdx);

            EncodeChannel(chnlData, waveletChnlData.get(), chnlIdx, compImage.channelData[chnlIdx]);

            if (waveletChnlData)
            {
                m_waveletCoeffs.EmplaceChannel(*waveletChnlData, chnlIdx);
            }
        };

#ifdef FLAIR_ENABLE_MULTITHREADING
        
        std::vector<std::thread> workerThreads;
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            workerThreads.emplace_back(EncodeChannelFunctor, chnlIdx);
            Assert(workerThreads.back().joinable());
        }
        for (auto& t : workerThreads) { t.join(); }

#else
        // Encode the image channel by channel
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            EncodeChannelFunctor(chnlIdx);
        }
#endif
    }

    //using DiagnosticWavelet = StaticDWT1<CDF97<float>>;
    using DiagnosticWavelet = StaticDWT2<float>;

    void Codec::EncodeWaveletCoeffs(const Image3f& inputImage, Image3f& waveletCoeffs) const
    {
        waveletCoeffs.Resize(inputImage);
        Image3f remappedImage = inputImage;

        // Transform the image to gamma-corrected space
        remappedImage.ApplyGamma(1. / 2.2);

        // Transform into YUV colour space
        remappedImage.RGBToYUV();

        // Encode the image channel by channel
        Image1f chnlData(remappedImage.Width(), remappedImage.Height());
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            Image1f chnlData = remappedImage.ExtractChannel(chnlIdx);

            DiagnosticWavelet dwt(m_width);
            dwt.Forward(chnlData.Vector());

            waveletCoeffs.EmplaceChannel(chnlData, chnlIdx);
        }
    }

    void Codec::DecodeWaveletCoeffs(const Image3f& waveletCoeffs, Image3f& outputImage) const
    {
        outputImage.Resize(waveletCoeffs);
        
        // Encode the image channel by channel
        Image1f chnlData(waveletCoeffs.Width(), waveletCoeffs.Height());
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            Image1f chnlData = waveletCoeffs.ExtractChannel(chnlIdx);

            DiagnosticWavelet dwt(m_width);
            dwt.Inverse(chnlData.Vector());
            
            outputImage.EmplaceChannel(chnlData, chnlIdx);
        }

        // Transform back into RGB colour space
        outputImage.YUVToRGB();

        // Transform the image to gamma-corrected space
        outputImage.ApplyGamma(2.2);
    }

    void Codec::Decode(const CompressedImageData& compImage, Image3f& outputImage)
    {
        PrepareDecoder(compImage);       

        outputImage.Resize(m_width, m_height);

        // Functor that decodes a channel then emplaces it in the decompressed image
        std::function<void(int)> DecodeChannelFunctor = [&, this](int chnlIdx)
        {
            const Image1f chnlData = DecodeChannel(compImage.channelData[chnlIdx], chnlIdx);    

            outputImage.EmplaceChannel(chnlData, chnlIdx);
        };

#ifdef FLAIR_ENABLE_MULTITHREADING

        std::vector<std::thread> workerThreads;
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            workerThreads.emplace_back(DecodeChannelFunctor, chnlIdx);
            Assert(workerThreads.back().joinable());
        }
        for (auto& t : workerThreads) { t.join(); }

#else
        // Encode the image channel by channel
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            DecodeChannelFunctor(chnlIdx);
        }
#endif

        // Transform back into RGB colour space
        outputImage.YUVToRGB();

        // Transform the image to gamma-corrected space
        outputImage.ApplyGamma(m_params.imageGamma);
    }
}
