#pragma once

#include "../Includes.h"
#include "../math/MathUtils.h"
#include <functional>

#if !defined(FLAIR_DISABLE_MULTITHREADING)
#include <thread>
#endif

namespace Flair
{   
    struct ImageRect
    {
        int x0, y0, x1, y1;

        ImageRect() : x0(std::numeric_limits<int>::max()), y0(std::numeric_limits<int>::max()), x1(std::numeric_limits<int>::min()), y1(std::numeric_limits<int>::min()) {}
        ImageRect(int _x0, int _y0, int _x1, int _y1) : x0(_x0), y0(_y0), x1(_x1), y1(_y1) {}

        inline int Area() const { return (x1 - x0) * (y1 - y0); }
        inline int Width() const { return x1 - x0; }
        inline int Height() const { return y1 - y0; }
        inline operator bool() const { return x1 > x0 && y1 > y0; }
        inline bool Contains(const int x, const int y) const { return x >= x0 && x < x1&& y >= y0 && y < y1; }
    };

    enum ImageFlags : int { kImageNearest, kImageBilinear };

    inline ImageRect Intersection(const ImageRect& a, const ImageRect& b)
    {
        return ImageRect(std::max(a.x0, b.x0), std::max(a.y0, b.y0), std::min(a.x1, b.x1), std::min(a.y1, b.y1));
    }

    template<typename Type, int Channels>
    class Image
    {
    public:
        using MapFunctor = std::function<void(int, int, Type*)>;
        using ParallelMapFunctor = std::function<void(int, int, int, Type*)>;

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

        template<int OtherChannels>
        inline void Resize(const Image<Type, OtherChannels>& other) { Resize(other.Width(), other.Height()); }

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
            AssertFmt(chnlData.Width() == m_width && chnlData.Height() == m_height, "Size mismatch!");
            for (int i = 0; i < m_area; ++i)
            {
                m_data[i * Channels + chnlIdx] = chnlData[i];
            }
        }
   
        operator bool() const { return !m_data.empty(); }
        bool Contains(const int x, const int y) const { return x >= 0 && x < m_width&& y >= 0 && y < m_height; }

        inline int Width() const { return m_width; }
        inline int Height() const { return m_height; }
        inline int Area() const { return m_area; }
        inline int Size() const { return m_area * Channels; }
        inline ImageRect Rect() const { return ImageRect(0, 0, m_width, m_height); }

        inline Type* operator()(const int x, const int y) { return &m_data[(y * m_width + x) * Channels]; }
        inline const Type* operator()(const int x, const int y) const { return &m_data[(y * m_width + x) * Channels]; }
        inline Type* At(const int x, const int y) { return &m_data[(y * m_width + x) * Channels]; }
        inline const Type* At(const int x, const int y) const { return &m_data[(y * m_width + x) * Channels]; }
        inline Type& operator[](const int i) { return m_data[i]; }
        inline Type operator[](const int i) const { return m_data[i]; }

        template<int InterpolationType>
        void Sample(float u, float v, Type* pixel) const
        {
            if (InterpolationType == kImageBilinear)
            {
                // For interpolation, we assume that the pixel values are defined at the mid-point of each logcal pixel
                // and that the values are clamped at the boundaries.
                //  
                // Example: for a 2 pixel image in 1 dimensions, values p[0] and p[1] correspond to coordinates 0.25 and 0.75 respectively
                // 
                // 0.0      0.5       1.0
                //  |   *    |    *    |
                //     p[0]     p[1]
                 
                int iu, iv;
                float du, dv;
                u = std::max(0.f, u * m_width - 0.5f);
                v = std::max(0.f, v * m_height - 0.5f);
                if (u >= m_width - 1) { iu = m_width - 2; du = 1; }
                else { iu = int(u); du = fract(u); }
                if (v >= m_height - 1) { iv = m_height - 2; dv = 1; }
                else { iv = int(v); dv = fract(v); }

                int idx = (iv * m_width + iu) * Channels;
                for (int c = 0; c < Channels; ++c)
                {
                    const Type t00 = m_data[idx + c];
                    const Type t10 = m_data[idx + Channels + c];
                    const Type t01 = m_data[idx + m_width * Channels + c];
                    const Type t11 = m_data[idx + (m_width + 1) * Channels + c];
                    pixel[c] = mix(mix(t00, t10, du), mix(t01, t11, du), dv);
                }
            }
            if (InterpolationType == kImageNearest)
            {
                int idx = Channels * (clamp(int(v * m_height - 0.5), 0, m_height - 1) * m_width +
                                      clamp(int(u * m_width - 0.5), 0, m_width - 1));

                for (int c = 0; c < Channels; ++c, ++idx)
                {
                    pixel[c] = m_data[idx];
                }
            }
        }

        template<int InterpolationType>
        inline Type Sample(float u, float v) const
        {
            Type pixel[Channels];
            Sample<InterpolationType>(u, v, pixel);
            return pixel[0];
        }

        inline void Sample(int x, int y, Type* pixel) const
        {
            const int idx = Channels * (clamp(y, 0, m_height - 1) * m_width + clamp(x, 0, m_width - 1));
            for (int c = 0; c < Channels; ++c, ++idx)
            {
                pixel[c] = m_data[idx];
            }
        }

        inline Type Sample(int x, int y) const
        {
            return m_data[Channels * (clamp(y, 0, m_height - 1) * m_width + clamp(x, 0, m_width - 1))];
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

        // Sets all pixels in the image to zero
        void Erase() { std::memset(m_data.data(), 0, sizeof(Type) * m_data.size()); }

        void Saturate() { for (auto& p : m_data) { p = saturate(p); } }

        void Clamp(const Type lower, const Type upper) { for (auto& p : m_data) { p = clamp(p, lower, upper); } }

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

        int GetThreadCount(const int maxThreads = 16) const
        {
#if defined(FLAIR_DISABLE_MULTITHREADING)
            return 1;
#else
            int numThreads = std::max(1, int(std::thread::hardware_concurrency()));
            if (maxThreads > 0)
            {
                numThreads = std::min(maxThreads, numThreads);
            }
            return numThreads;
#endif
        }

        void ParallelMap(ParallelMapFunctor setPixel, ImageRect region = ImageRect(), const int maxThreads = 16)
        {
            // If no region was specified, reinitialise it to the entire image
            if (!region) { region = ImageRect(0, 0, m_width, m_height); }

#if defined(FLAIR_DISABLE_MULTITHREADING)

            for (int y = region.y0; y < region.y1; ++y)
            {
                for (int x = region.x0; x < region.x1; ++x)
                {
                    setPixel(x, y, 0, &m_data[Channels * (y * m_width + x)]);
                }
            }
#else
            const int numThreads = GetThreadCount(maxThreads);
            const int regionArea = region.Area();

            // Launch the worker threads
            std::vector<std::thread> workers;
            for (int i = 0; i < numThreads; ++i)
            {
                const int startPixel = i * regionArea / numThreads;
                const int endPixel = (i + 1) * regionArea / numThreads;
                workers.emplace_back(&Image<Type, Channels>::MapThread, this, region, startPixel, endPixel, i, setPixel);
            }

            // Wait for all the workers to finish
            for (int i = 0; i < numThreads; ++i) { workers[i].join(); }
#endif
        }

        void Map(MapFunctor setPixel, ImageRect region = ImageRect(), const int maxThreads = 16)
        {
            // If no region was specified, reinitialise it to the entire image
            if (!region) { region = ImageRect(0, 0, m_width, m_height); }

            for (int y = region.y0; y < region.y1; ++y)
            {
                for (int x = region.x0; x < region.x1; ++x)
                {
                    setPixel(x, y, &m_data[Channels * (y * m_width + x)]);
                }
            }
        }

    private:
        void MapThread(const ImageRect& region, const int startPixel, const int endPixel, const int threadIdx, ParallelMapFunctor setPixel)
        {
            for (int i = startPixel; i < endPixel; ++i)
            {
                const int x = region.x0 + i % region.Width();
                const int y = region.y0 + i / region.Width();
                setPixel(x, y, threadIdx, &m_data[Channels * (y * m_width + x)]);
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