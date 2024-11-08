#pragma once

#include "../Includes.h"
#include <functional>

#ifdef FLAIR_ENABLE_MULTITHREADING
#include <thread>
#endif

namespace Flair
{   
    template<typename Type, int Channels>
    class Image
    {
    public:
        using PopulateFunctor = std::function<void(int, int, Type*)>;

    public:
        Image() : Image(0, 0) {}
        
        Image(const int width, const int height, const Type* data = nullptr) : 
            m_width(0),
            m_height(0),
            m_area(0)
        {
            Resize(width, height);
            if (data)
            {
                memcpy(m_data.data(), data, sizeof(Type) * Channels * m_area);
            }
        }

        Image(const Image& other)
        {
            *this = other;
        }

        Image& operator=(const Image& other)
        {
            m_data = other.m_data;
            m_width = other.m_width;
            m_height = other.m_height;
            m_area = other.m_area;
            return *this;
        }

        void Resize(const int width, const int height)
        {
            if (m_width == width && m_height == height) { return; }
            
            AssertFmt(width >= 0 && height >= 0, "Invalid image dimensions: %i x %i", width, height);
            m_width = width;
            m_height = height;
            m_area = width * height;
            
            m_data.clear();
            m_data.resize(m_area * Channels, Type(0));
        }

        Image<Type, 1> ExtractChannel(const int chnlIdx) const
        {
            Image<Type, 1> chnlData(m_width, m_height);
            for (int i = 0; i < m_area; ++i)
            {
                chnlData[i] = m_data[i * Channels + chnlIdx];
            }
            return chnlData;
        }       

        Image<Type, 1> ExtractLuminance() const
        {
            static_assert(Channels == 3, "Extract luminance requires a 3-channel RGB image.");
            Image<Type, 1> lum(m_width, m_height);
            for (int i = 0, j = 0; i < m_area; ++i, j += 3)
            {
                Assert(i < lum.Vector().size());
                lum[i] = m_data[j] * 0.17691 + m_data[j + 1] * 0.8124 + m_data[j + 2] * 0.01063;
            }
            return lum;
        }

        void EmplaceChannel(const Image<Type, 1>& chnlData, const int chnlIdx)
        {
            AssertMsg(chnlData.Width() == m_width && chnlData.Height() == m_height, "Size mismatch!");
            for (int i = 0; i < m_area; ++i)
            {
                m_data[i * Channels + chnlIdx] = chnlData[i];
            }
        }

        inline void Resize(const Image& other) { Resize(other.Width(), other.Height()); }

        operator bool() const { return !m_data.empty(); }
        bool Contains(const int x, const int y) const { return x >= 0 && x < m_width&& y >= 0 && y < m_height; }

        inline int Width() const { return m_width; }
        inline int Height() const { return m_height; }
        inline int Area() const { return m_area; }
        inline int Size() const { return m_area * Channels; }

        inline Type* operator()(const int x, const int y) { return &m_data[(y * m_width + x) * Channels]; }
        inline const Type* operator()(const int x, const int y) const { return &m_data[(y * m_width + x) * Channels]; }      
        inline Type* At(const int x, const int y) { return &m_data[(y * m_width + x) * Channels]; }
        inline const Type* At(const int x, const int y) const { return &m_data[(y * m_width + x) * Channels]; }
        inline Type& operator[](const int i) { return m_data[i]; }
        inline Type operator[](const int i) const { return m_data[i]; }

        void Bilinear(float u, float v, float* pixel) const
        {
            int iu, iv;
            float du, dv;
            u = std::max(0.f, u * (m_width - 1));
            v = std::max(0.f, v * (m_height - 1));
            if (u >= m_width - 1) { iu = m_width - 2; du = 1; }
            else { iu = int(u); du = fract(u); }
            if (v >= m_height - 1) { iv = m_height - 2; dv = 1; }
            else { iv = int(v); dv = fract(v); }

            int idx = (iv * m_width + iu) * Channels;
            for (int c = 0; c < Channels; ++c)
            {
                const float t00 = m_data[idx+c];
                const float t10 = m_data[idx+Channels+c];
                const float t01 = m_data[idx+m_width*Channels+c];
                const float t11 = m_data[idx+(m_width+1)*Channels+c];
                pixel[c] = mix(mix(t00, t10, du), mix(t01, t11, du), dv);
            }
        }

        void NearestNeighbour(float u, float v, float* pixel) const
        {
            int idx = 3 * (clamp(int(v * (m_height - 1)), 0, m_height - 1) * m_width + 
                           clamp(int(u * (m_width - 1)), 0, m_width - 1));

            for (int c = 0; c < Channels; ++c, ++idx)
            {
                pixel[c] = m_data[idx];
            }
        }

        void ApplyGamma(const float gamma)
        {
            if (gamma != 1)
            {
                for (auto& p : m_data)
                {
                    p = std::pow(p, gamma);
                }
            }
        }

        void Erase()
        {
            for (auto& p : m_data) { p = 0; }
        }

        void RGBToYUV()
        {
            static_assert(Channels == 3, "RGBToYUV requires 3-channel image");
            
            Type yuv[3];
            Type* rgb = m_data.data();
            for (int i = 0; i < m_area; ++i, rgb += 3)
            {
                yuv[0] = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114;
                yuv[1] = rgb[0] * -0.14713 + rgb[1] * -0.28886 + rgb[2] * 0.436;
                yuv[2] = rgb[0] * 0.615 + rgb[1] * -0.51499 + rgb[2] * -0.10001;
                memcpy(rgb, yuv, sizeof(Type) * 3);
            }
        }

        void YUVToRGB()
        {
            static_assert(Channels == 3, "YUVToRGB requires 3-channel image");
            
            Type rgb[3];
            Type* yuv = m_data.data();
            for (int i = 0; i < m_area; ++i, yuv += 3)
            {
                rgb[0] = yuv[0] * 1. + yuv[1] * 0. + yuv[2] * 1.13983;
                rgb[1] = yuv[0] * 1 + yuv[1] * -0.394565 + yuv[2] * -0.58060;
                rgb[2] = yuv[0] * 1 + yuv[1] * 2.03211 + yuv[2] * 0;
                memcpy(yuv, rgb, sizeof(Type) * 3);
            }
        }

        Type* Data() { return m_data.data(); }
        const Type* Data() const { return m_data.data(); }
        std::vector<Type>& Vector() { return m_data; }
        const std::vector<Type> Vector() const { return m_data; }

        template<typename Lambda>
        void Populate(Lambda setPixel)
        {
            for (int y = 0, i = 0; y < m_height; ++y)
            {
                for (int x = 0; x < m_width; ++x, i += 3)
                {
                    setPixel(x, y, &m_data[i]);
                }
            }
        }

#ifdef FLAIR_ENABLE_MULTITHREADING

        void PopulateParallel(PopulateFunctor setPixel, const int maxThreads = 16)
        {
            int numThreads = std::max(1, int(std::thread::hardware_concurrency()));
            if (maxThreads > 0)
            {
                numThreads = std::min(maxThreads, numThreads);
            }
            
            // Launch the worker threads
            std::vector<std::thread> workers;
            for (int i = 0; i < numThreads; ++i)
            {
                const int startPixel = i * m_area / numThreads;
                const int endPixel = (i + 1) * m_area / numThreads;
                workers.emplace_back(&Image<Type, Channels>::PopulateThread, this, startPixel, endPixel, setPixel);
            }

            // Wait for all the workers to finish
            for (int i = 0; i < numThreads; ++i) { workers[i].join(); }
        }

#else

        inline void PopulateParallel(PopulateFunctor setPixel, const int maxThreads = 0) { Populate(setPixel); }

#endif

    private:
        void PopulateThread(const int startPixel, const int endPixel, PopulateFunctor setPixel)
        {
            for (int i = startPixel; i < endPixel; ++i)
            {
                setPixel(i % m_width, i / m_width, &m_data[i*3]);
            }
        }

    private:
        std::vector<Type>   m_data;
        int                 m_width;
        int                 m_height;
        int                 m_area;
    };

    using Image3f = Image<float, 3>;
    using Image1f = Image<float, 1>;
}