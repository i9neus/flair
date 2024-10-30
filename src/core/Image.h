#pragma once

#include "Includes.h"

namespace HDRI
{
    template<typename Type, size_t Channels>
    class Image
    {
    public:
        Image() : Image(0, 0) {}
        
        Image(const int width, const int height, const Type* data = nullptr)
        {
            Resize(width, height);
            if (data)
            {
                memcpy(m_pixels.data(), data, sizeof(Type) * m_area);
            }
        }

        Image& operator=(const Image& other)
        {
            m_pixels = other.m_pixels;
            m_width = other.m_width;
            m_height = other.m_height;
            m_area = other.m_area;
        }

        void Resize(const int width, const int height)
        {
            if (m_width == width && m_height == height) { return; }
            
            AssertFmt(width >= 0 && height >= 0, "Invalid image dimensions: %i x %i", width, height);
            m_width = width;
            m_height = height;
            m_area = width * height;
            
            m_pixels.clear();
            m_pixels.resize(m_area * Channels, Type(0));
        }

        void ExtractChannel(Image<Type, 1>& chnlData, const int chnlIdx) const
        {
            chnlData.Resize(m_width, m_height);
            for (int i = 0; i < m_area; ++i)
            {
                chnlData[i] = m_pixels[i * Channels + chnlIdx];
            }
        }

        void EmplaceChannel(const Image<Type, 1>& chnlData, const int chnlIdx)
        {
            AssertMsg(chnlData.Width() == m_width && chnlData.Height() == m_height, "Size mismatch!");
            for (int i = 0; i < m_area; ++i)
            {
                m_pixels[i * Channels + chnlIdx] = chnlData[i];
            }
        }

        inline void Resize(const Image& other) { Resize(other.Width(), other.Height()); }

        operator bool() const { return !m_pixels.empty(); }
        bool Contains(const int x, const int y) const { return x >= 0 && x < m_width&& y >= 0 && y < m_height; }

        template<typename Lambda>
        void Populate(Lambda setPixel)
        {
            for (int y = 0, i = 0; y < m_height; ++y)
            {
                for (int x = 0; x < m_width; ++x, i += Channels)
                {
                    setPixel(x, y, &m_pixels[i]);
                }
            }
        }

        inline int Width() const { return m_width; }
        inline int Height() const { return m_height; }
        inline int Area() const { return m_area; }
        inline int Size() const { return m_area * Channels; }

        inline Type* operator()(const int x, const int y) { return &m_pixels[(y * m_width + x) * Channels]; }
        inline const Type* operator()(const int x, const int y) const { return &m_pixels[(y * m_width + x) * Channels]; }      
        inline Type* At(const int x, const int y) { return &m_pixels[(y * m_width + x) * Channels]; }
        inline const Type* At(const int x, const int y) const { return &m_pixels[(y * m_width + x) * Channels]; }
        inline Type& operator[](const int i) { return m_pixels[i]; }
        inline Type operator[](const int i) const { return m_pixels[i]; }

        void ApplyGamma(const float gamma)
        {
            if (gamma != 1)
            {
                for (auto& p : m_pixels)
                {
                    p = std::pow(p, gamma);
                }
            }
        }

        void Erase()
        {
            for (auto& p : m_pixels) { p = 0; }
        }

        template<typename = typename std::enable_if_t<Channels == 3>>
        void RGBToYUV()
        {
            Type yuv[3];
            Type* rgb = m_pixels.data();
            for (int i = 0; i < m_area; ++i, rgb += 3)
            {
                yuv[0] = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114;
                yuv[1] = rgb[0] * -0.14713 + rgb[1] * -0.28886 + rgb[2] * 0.436;
                yuv[2] = rgb[0] * 0.615 + rgb[1] * -0.51499 + rgb[2] * -0.10001;
                memcpy(rgb, yuv, sizeof(Type) * 3);
            }
        }

        template<typename = typename std::enable_if_t<Channels == 3>>
        void YUVToRGB()
        {
            Type rgb[3];
            Type* yuv = m_pixels.data();
            for (int i = 0; i < m_area; ++i, yuv += 3)
            {
                rgb[0] = yuv[0] * 1. + yuv[1] * 0. + yuv[2] * 1.13983;
                rgb[1] = yuv[0] * 1 + yuv[1] * -0.394565 + yuv[2] * -0.58060;
                rgb[2] = yuv[0] * 1 + yuv[1] * 2.03211 + yuv[2] * 0;
                memcpy(yuv, rgb, sizeof(Type) * 3);
            }
        }

        Type* Data() { return m_pixels.data(); }
        const Type* Data() const { return m_pixels.data(); }
        std::vector<Type>& Vector() { return m_pixels; }
        const std::vector<Type> Vector() const { return m_pixels; }

    private:
        std::vector<Type>   m_pixels;
        int                 m_width;
        int                 m_height;
        int                 m_area;
    };

    using Image3f = Image<float, 3>;
    using Image1f = Image<float, 1>;
}