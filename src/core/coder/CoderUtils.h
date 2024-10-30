#pragma once

#include "../Includes.h"

namespace HDRI
{
    template<typename T>
    static std::string bstr(const T& t, const bool truncate = false)
    {
        static_assert(std::is_integral<T>::value, "Not an integer");

        std::string str;
        for (int j = sizeof(t) * 8 - 1; j >= 0; j--)
        {
            str += (t & (T(1) << j)) ? '1' : '0';

            if (j != 0 && j % 8 == 0) { str += ','; }
        }
        return str;
    }

    template<typename T>
    static T btoi(const std::string in)
    {
        T out = 0;
        int j = 31;
        for (int i = 0; i < in.length() && j >= 0; i++)
        {
            if (in[i] != '0' && in[i] != '1') { continue; }

            out |= uint32_t(in[i] - '0') << j--;
        }
        return out;
    }

    template<typename OutputType, typename InputType>
    static OutputType Cast(const InputType& input)
    {
        OutputType output;
        output.resize(input.size());
        std::memcpy(output.data(), input.data(), input.size());
        return output;
    }

    inline std::vector<char> Cast(const char* input)
    {
        return Cast<std::vector<char>>(std::string(input));
    }

    template<typename T>
    static uint32_t GetPackedMessageLength(const std::vector<T>& packed)
    {
        if (packed.empty()) { return 0; }

        constexpr uint32_t typeSize = sizeof(T) * 8;
        uint32_t messageLength = typeSize * (packed.size() - 1);
        for (int bit = 0; bit < typeSize; ++bit)
        {
            if (packed.back() & (T(1) << bit))
            {
                messageLength += typeSize - bit;
                break;
            }
        }

        return messageLength;
    }

    template<typename T>
    static std::string packed_bstr(const std::vector<T>& packed, int start = -1, int end = -1)
    {
        if (packed.empty()) { return ""; }

        if (start == -1) { start = 0; }
        if (end == -1 || end > packed.size()) { end = packed.size(); }

        // Find the least-significant non-zero bit
        int j = sizeof(T) * 8 * start;
        std::string str;
        const uint32_t messageLength = GetPackedMessageLength(packed);

        for (int idx = start; idx < end; idx++)
        {
            const auto& element = packed[idx];

            for (int bit = sizeof(T) * 8 - 1; bit >= 0 && j < messageLength; --bit, j++)
            {
                if (j != 0 && j % 8 == 0) { str += ','; }

                str += (element & (T(1) << bit)) ? '1' : '0';
            }
        }
        return str;
    }

    // Simple Xor-based hash for verifying data integrity
    static unsigned char XorHash8(const unsigned char* data, const size_t dataSize)
    {
        if (dataSize == 0) { return 0; }

        unsigned char hash = data[0];
        for (int i = 1; i < dataSize; ++i)
        {
            hash ^= data[i] + 149;
        }
        return hash;
    }

    template<typename T>
    inline unsigned char XorHash8(const std::vector<T>& input)
    {
        const unsigned char* data = reinterpret_cast<const unsigned char*>(input.data());
        return XorHash8(data, data.size() * sizeof(T));
    }
}