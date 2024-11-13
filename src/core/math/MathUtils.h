#pragma once

#include <limits>
#include <cmath>
#include <cstdint>

namespace Flair
{
    constexpr float kPi = 3.141592653589793f;
    constexpr float kTwoPi = 2 * kPi;
    constexpr float kFourPi = 4 * kPi;
    constexpr float kHalfPi = 0.5 * kPi;
    constexpr float kRoot2 = 1.4142135623730951f;
    constexpr float kFltMax = std::numeric_limits<float>::max();
    constexpr float kPhi = 1.6180339887498948f;
    constexpr float kInvPhi = 1 / kPhi;
    constexpr float kLog2 = 0.6931471805599453f;

#define kXAxis      Vec3(1, 0, 0)
#define kYAxis      Vec3(0, 1, 0)
#define kZAxis      Vec3(0, 0, 1)

    template<typename T> inline T sign(const T f) { return std::copysign(T(1), f); }

    inline float toRad(float deg) { return kTwoPi * deg / 360; }
    inline float toDeg(float rad) { return 360 * rad / kTwoPi; }
    template<typename T> inline T sqr(const T t) { return t * t; }
    template<typename T> inline T cub(const T t) { return t * t * t; }
    template<typename T> inline T pow4(T t) { t *= t; return t * t; }

    // Complement modulus. Negative values are wrapped around to become positive values. e.g. -2 % 10 = 8
    inline int compMod(int a, int b) { return ((a % b) + b) % b; }

    // Heaviside step function
    inline float heaviside(const float edge, const float t) { return float(t > edge); }

    // Clamp value in the range [a, b]
    template<typename T> T clamp(const T v, const T a, const T b) { return ((v < a) ? a : ((v > b) ? b : v)); }

    // Clamp value in the range [0, 1]
    inline float saturate(const float v) { return clamp(v, 0.f, 1.f); }

    // Return the fractional component of a f
    inline float fract(const float f) { return std::fmod(f, 1.0f); }

    // Lerp between a and b with parameter t
    template<typename T, typename S>
    inline S mix(const S& a, const S& b, const T& t) { return a * (1 - t) + b * t; }

    template<typename T, typename S>
    inline S smoothstep(const S& a, const S& b, const T& t) { return mix(a, b, t * t * (3 - 2 * t)); }

    inline float smoothstep(const float& t) { return mix(0.f, 1.f, t); }

    // Maps t in the range [0, 1] onto a cosine curve
    inline float trigInterpolate(const float t) { return std::cos(t * kPi + kPi) * 0.5f + 0.5f; }

    // Maps t in the range [0, 1] onto a cosine curve with an exponential fall-off towards the extrema
    inline float trigInterpolateExp(const float t, const float ex)
    {
        float s = std::abs(t * 2 - 1);
        s = std::sin(std::pow(s, ex) * kHalfPi);
        return (t < 0.5f) ? ((1 - s) * 0.5f) : (0.5f + 0.5f * s);
    }

    // Maps t in the range [0, 1] onto a sigmoid curve with slope defined by sigma
    inline float sigmoidInterpolate(const float t, const float sigma)
    {
        const float residue = 1 / (1 + std::exp(-sigma));
        float f = 1 / (1 + std::exp(-(t * 2 - 1) * sigma));
        return (f - residue) / (1 - 2 * residue);
    }

    // FNV1a hash of an array of bytes
    inline uint32_t HashOf(const char* data, const size_t numBytes)
    {
        uint32_t hash = 0x811c9dc5u;
        for (int i = 0; i < numBytes; ++i)
        {
            hash = (hash ^ data[i]) * 0x01000193u;
        }
        return hash;
    }

    // Mix and combine two hashes
    inline uint32_t HashCombine(const uint32_t a, const uint32_t b)
    {
        return (((a << (31u - (b & 31u))) | (a >> (b & 31u)))) ^
            ((b << (a & 31u)) | (b >> (31u - (a & 31u))));
    }

}
