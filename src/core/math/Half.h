#pragma once

#include <cstdint>

namespace Flair
{
    inline uint16_t FloatToHalfBits(const float f32)
    {
        union { uint32_t u; float f; };
        f = f32;

        // Handle zero as special case
        if (u == 0u)
        {
            return 0u;
        }
        else
        {
            uint32_t expo = (u >> 23) & 0xffu;
            if (expo < 127u - 15u) { expo = 0u; } // Underflow
            else if (expo > 127u + 16u) { expo = 31u; } // Overflow
            else { expo = ((u >> 23) & 0xffu) + 15u - 127u; } // Biased exponent

            // Composite
            return ((u >> 16) & (1u << 15)) |  // Sign bit
                ((u & ((1u << 23) - 1u)) >> 13) | // Fraction
                ((expo & ((1u << 5) - 1u)) << 10); // Exponent
        }
    }

    inline float HalfBitsToFloat(const uint16_t u16)
    {
        if (u16 == 0u) { return 0.; }

        union { uint32_t u; float f; };
        u = u16;

        u = ((u & (1u << 15)) << 16) | // Sign bit 
            ((u & ((1u << 10u) - 1u)) << 13) | // Fraction
            ((((u >> 10) & ((1u << 5) - 1u)) + 127u - 15u) << 23); // Exponent

        return f;
    }
}