#pragma once

#include <limits>
#include <cmath>
#include <cstdint>

#if !defined(__host__) 
#define __host__
#endif
#if !defined(__device__) 
#define __device__
#endif
#if !defined(__forceinline__) 
#define __forceinline__ inline
#endif

namespace Flair
{   
    #define kXAxis      Vec3(1, 0, 0)
    #define kYAxis      Vec3(0, 1, 0)
    #define kZAxis      Vec3(0, 0, 1)
    
    static constexpr float kPi          = 3.141592653589793f;
    static constexpr float kTwoPi       = 2 * kPi;
    static constexpr float kFourPi      = 4 * kPi;
    static constexpr float kHalfPi      = 0.5 * kPi;
    static constexpr float kRoot2       = 1.4142135623730951f;
    static constexpr float kFltMax      = std::numeric_limits<float>::max();
    static constexpr float kPhi         = 1.6180339887498948f;
    static constexpr float kInvPhi      = 1 / kPhi;
    static constexpr float kLog2        = 0.6931471805599453f;

    template<typename T> __forceinline__  __host__ __device__ T sign(const T f) 
    {
        //return std::copysign(T(1), f);
        return 1 - 2 * T(f < 0);
    }

    __host__ __device__ __forceinline__ float  __host__ __device__ toRad(float deg) { return kTwoPi * deg / 360; }
    __host__ __device__ __forceinline__ float  __host__ __device__ toDeg(float rad) { return 360 * rad / kTwoPi; }
    template<typename T> __host__ __device__ __forceinline__ T sqr(const T t) { return t * t; }
    template<typename T> __host__ __device__ __forceinline__ T cub(const T t) { return t * t * t; }
    template<typename T> __host__ __device__ __forceinline__ T pow4(T t) { t *= t; return t * t; }

    // Complement modulus. Negative values are wrapped around to become positive values. e.g. -2 % 10 = 8
    __host__ __device__ __forceinline__ int compMod(int a, int b) { return ((a % b) + b) % b; }

    // Heaviside step function
    __host__ __device__ __forceinline__ float heaviside(const float edge, const float t) { return float(t > edge); }

    // Clamp value in the range [a, b]
    template<typename T> __host__ __device__ __forceinline__ T clamp(const T v, const T a, const T b) { return ((v < a) ? a : ((v > b) ? b : v)); }

    // Clamp value in the range [0, 1]
    __host__ __device__ __forceinline__ float saturate(const float v) { return clamp(v, 0.f, 1.f); }

    // Return the fractional component of a f
    __host__ __device__ __forceinline__ float fract(const float f) { return std::fmod(f, 1.0f); }

    // Lerp between a and b with parameter t
    template<typename T, typename S>
    __host__ __device__ __forceinline__ S mix(const S& a, const S& b, const T& t) { return a * (1 - t) + b * t; }

    template<typename T, typename S>
    __host__ __device__ __forceinline__ S smoothstep(const S& a, const S& b, const T& t) { return mix(a, b, t * t * (3 - 2 * t)); }

    __host__ __device__ __forceinline__ float smoothstep(const float& t) { return mix(0.f, 1.f, t); }

    // Maps t in the range [0, 1] onto a cosine curve
    __host__ __device__ __forceinline__ float trigInterpolate(const float t) { return std::cos(t * kPi + kPi) * 0.5f + 0.5f; }

    // Maps t in the range [0, 1] onto a cosine curve with an exponential fall-off towards the extrema
    __host__ __device__ __forceinline__ float trigInterpolateExp(const float t, const float ex)
    {
        float s = std::abs(t * 2 - 1);
        s = std::sin(std::pow(s, ex) * kHalfPi);
        return (t < 0.5f) ? ((1 - s) * 0.5f) : (0.5f + 0.5f * s);
    }

    // Maps t in the range [0, 1] onto a sigmoid curve with slope defined by sigma
    __host__ __device__ __forceinline__ float sigmoidInterpolate(const float t, const float sigma)
    {
        const float residue = 1 / (1 + std::exp(-sigma));
        float f = 1 / (1 + std::exp(-(t * 2 - 1) * sigma));
        return (f - residue) / (1 - 2 * residue);
    }

    // FNV1a hash of an array of bytes
    __host__ __device__ __forceinline__ uint32_t HashOf(const char* data, const size_t numBytes)
    {
        uint32_t hash = 0x811c9dc5u;
        for (int i = 0; i < numBytes; ++i)
        {
            hash = (hash ^ data[i]) * 0x01000193u;
        }
        return hash;
    }

    // Mix and combine two hashes
    __host__ __device__ __forceinline__ uint32_t HashCombine(const uint32_t a, const uint32_t b)
    {
        return (((a << (31u - (b & 31u))) | (a >> (b & 31u)))) ^
            ((b << (a & 31u)) | (b >> (31u - (a & 31u))));
    }

    // Reverse bits of 32-bit integer
    __host__ __device__ __forceinline__ uint32_t RadicalInverse(uint32_t i)
    {
        i = ((i & 0xffffu) << 16u) | (i >> 16u);
        i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
        i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
        i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);
        i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);
        return i;
    }

    template<typename T>
    __host__ __device__ __forceinline__ void Swap(T& a, T& b)
    {
        const T s = a;
        a = b;
        b = s;
    }

}
