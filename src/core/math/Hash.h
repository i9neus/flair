#pragma once

#include "../Includes.h"
#include <stdint.h>

namespace Flair
{
    class PCG
    {
    private:
        std::array<uint32_t, 4> m_state;

        void Advance()
        {
            m_state[0] = m_state[0] * 1664525u + 1013904223u;
            m_state[1] = m_state[1] * 1664525u + 1013904223u;
            m_state[2] = m_state[2] * 1664525u + 1013904223u;
            m_state[3] = m_state[3] * 1664525u + 1013904223u;

            m_state[0] += m_state[1] * m_state[3];
            m_state[1] += m_state[2] * m_state[0];
            m_state[2] += m_state[0] * m_state[1];
            m_state[3] += m_state[1] * m_state[2];

            m_state[0] ^= m_state[0] >> 16u;
            m_state[1] ^= m_state[1] >> 16u;
            m_state[2] ^= m_state[2] >> 16u;
            m_state[3] ^= m_state[3] >> 16u;

            m_state[0] += m_state[1] * m_state[3];
            m_state[1] += m_state[2] * m_state[0];
            m_state[2] += m_state[0] * m_state[1];
            m_state[3] += m_state[1] * m_state[2];
        }

    public:
        PCG(const uint32_t seed)
        {
            m_state[0] = 20219u * seed;
            m_state[1] = 7243u * seed;
            m_state[2] = 12547u * seed;
            m_state[3] = 28573u * seed;
        }

        // Generate tuples of canonical random numbers
        inline float Rand() { Advance(); return float(m_state[0]) / float(0xffffffffu); }
        inline std::tuple<float, float> Rand2() { Advance();  return { m_state[0] / float(0xffffffffu), m_state[1] / float(0xffffffffu) }; }
        inline std::tuple<float, float, float> Rand3() { Advance(); return { m_state[0] / float(0xffffffffu), m_state[1] / float(0xffffffffu), m_state[2] / float(0xffffffffu) }; }
        inline std::tuple<float, float, float, float> Rand4() { Advance(); return { m_state[0] / float(0xffffffffu), m_state[1] / float(0xffffffffu), m_state[2] / float(0xffffffffu), m_state[3] / float(0xffffffffu) }; }

        inline void RandN(float* rnd)
        {
            Advance();
            rnd[0] = m_state[0] / float(0xffffffffu);
            rnd[1] = m_state[1] / float(0xffffffffu);
            rnd[2] = m_state[2] / float(0xffffffffu);
            rnd[3] = m_state[3] / float(0xffffffffu);
        }
    };

    // Mix and combine hashes
    template<typename T>
    inline T HashCombine(const T& a, const T& b)
    {
        static_assert(std::is_integral<T>::value, "HashCombine requires integral type.");

        // My Shadertoy function
        return (((a << (31 - (b & 31))) | (a >> (b & 31)))) ^
            ((b << (a & 31)) | (b >> (31 - (a & 31))));

        // Based on Boost's hash_combine().
        // NOTE: Collides badly when hashing a string with only one letter difference
        //return b + 0x9e3779b9 + (a << 6) + (a >> 2);
    }

    // Mix and combine hashes
    template<typename T>
    inline uint32_t HashCombine(const std::hash<T>& a, const std::hash<T>& b)
    {
        return HashCombine(uint32_t(a), uint32_t(b));
    }

    inline uint32_t HashCombine(const float& a, const float& b)
    {
        return HashCombine(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
    }

    template<typename Type>
    inline uint32_t HashCombine(const Type& a) { return a; }

    template<typename Type, typename... Pack>
    inline uint32_t HashCombine(const Type& a, const Type& b, Pack... pack)
    {
        return HashCombine(HashCombine(a, b), HashCombine(pack...));
    }

    // Compute a 32-bit Fowler-Noll-Vo hash for the given input
    inline uint32_t HashOf(const uint32_t& i)
    {
        static constexpr uint32_t kFNVPrime = 0x01000193u;
        static constexpr uint32_t kFNVOffset = 0x811c9dc5u;

        uint32_t h = (kFNVOffset ^ (i & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 8u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 16u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 24u) & 0xffu)) * kFNVPrime;
        return h;
    }

    inline uint32_t HashOf(const float& i)
    {
        return HashOf(*reinterpret_cast<const uint32_t*>(&i));
    }

    inline uint32_t HashOf(const int& i)
    {
        return HashOf(*reinterpret_cast<const uint32_t*>(&i));
    }

    template<typename Type, typename... Pack>
    inline uint32_t HashOf(const Type& v0, const Pack&... pack)
    {
        return HashCombine(HashOf(v0), HashOf(pack...));
    }

    inline uint32_t HashOfArray(const char* data, const size_t numBytes)
    {
        uint32_t hash = 0x811c9dc5u;
        for (int i = 0; i < numBytes; ++i)
        {
            hash = (hash ^ data[i]) * 0x01000193u;
        }
        return hash;
    }

    template<typename... Pack>
    inline float HashOfAsFloat(const Pack&... pack)
    {
        auto h = HashOf(pack...);
        return float(h) / float(0xffffffff);
    }
    
    inline float OrderedDither(const int x, const int y)
    {
        static const float kOrderedDither[4][4] = { { 0.0, 8.0, 2.0, 10.}, {12., 4., 14., 6.}, { 3., 11., 1., 9.}, {15., 7., 13., 5.} };
        return (kOrderedDither[x & 3][y & 3] + 0.5) / 16.0;
    }
}