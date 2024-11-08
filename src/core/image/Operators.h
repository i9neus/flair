#pragma once

#include "Image.h"

namespace Flair
{
    /*template<typename Type, int Channels>
    Image<Type, 1> ExtractChannel(const Image<Type, Channels>& inputImg, const int chnlIdx)
    {
        Image<Type, 1> chnlImg(inputImg.Width(), inputImg.Height());
        for (int i = 0; i < inputImg.Area(); ++i)
        {
            chnlImg[i] = inputImg[i * Channels + chnlIdx];
        }
        return chnlImg;
    }

    template<typename Type, int Channels>
    Image<Type, 1> ExtractLuminance(const Image<Type, Channels>& inputImg)
    {
        static_assert(Channels == 3, "Extract luminance requires a 3-channel RGB image.");
        Image<Type, 1> lum(inputImg.Width(), inputImg.Height());
        for (int i = 0, j = 0; i < inputImg.Area(); ++i, j += 3)
        {
            lum[i] = inputImg[j] * 0.17691 + inputImg[j + 1] * 0.8124 + inputImg[j + 2] * 0.01063;
        }
        return lum;
    }

    template<typename Type, int Channels>
    void EmplaceChannel(Image<Type, Channels>& destImg, const Image<Type, 1>& chnlData, const int chnlIdx)
    {
        AssertMsg(destImg.Width() == chnlData.Width() && destImg.Height() == chnlData.Height(), "Size mismatch!");
        for (int i = 0; i < destImg.Area(); ++i)
        {
            destImg[i * Channels + chnlIdx] = chnlData[i];
        }
    }*/

    template<typename Type, int Channels>
    Image<Type, Channels> Downsample(Image<Type, Channels>& inputImg, int factor)
    {
        Image<Type, Channels> newImage(inputImg.Width() / factor, inputImg.Height() / factor);

        for (int y = 0, outIdx = 0; y < newImage.Height(); ++y)
        {
            for (int x = 0; x < newImage.Width(); ++x, outIdx += Channels)
            {
                const int u0 = x * inputImg.Width() / newImage.Width();
                const int u1 = (x + 1) * inputImg.Width() / newImage.Width();
                const int v0 = y * inputImg.Height() / newImage.Height();
                const int v1 = (y + 1) * inputImg.Height() / newImage.Height();
                int sumPixels = 0;

                float sigma[Channels] = {};
                for (int v = v0; v < v1; ++v)
                {
                    for (int u = u0; u < u1; ++u)
                    {
                        if (u >= 0 && u < inputImg.Width() && v >= 0 && v < inputImg.Height())
                        {
                            for (int c = 0; c < Channels; ++c)
                            {
                                sigma[c] += inputImg[(v * inputImg.Width() + u) * Channels + c];
                            }
                            ++sumPixels;
                        }
                    }

                    for (int c = 0; c < Channels; ++c) { newImage[outIdx + c] = sigma[c] / sumPixels; }
                }
            }
        }

        return newImage;
    }
}