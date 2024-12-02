#pragma once

#include "Image.h"

namespace Flair
{
    template<typename Type, int Channels>
    Image<Type, Channels> Downsample(const Image<Type, Channels>& inputImg, const int factor)
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

    template<typename Type, int Channels>
    Image<Type, Channels> Crop(const Image<Type, Channels>& inputImg, ImageRect cropRegion)
    {
        cropRegion = Intersection(inputImg.Rect(), cropRegion);

        Image<Type, Channels> newImage(cropRegion.Width(), cropRegion.Height());
        newImage.ParallelMap([&](const int x, const int y, const int i, float* outputPixel)
            {
                const float* inputPixel = inputImg.At(x + cropRegion.x0, y + cropRegion.y0);
                for (int c = 0; c < Channels; ++c)
                {
                    outputPixel[c] = inputPixel[c];
                }
            });
      
        return newImage;
    }
}