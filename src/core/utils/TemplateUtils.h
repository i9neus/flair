#pragma once

namespace Flair
{
    // Utility for retrieving the last parameter in a variadic pack
    template<typename... Pack>
    struct LastOf
    {
        template<typename T> struct Helper { using Type = T; };
        using Type = typename decltype((Helper<Pack>{}, ...))::Type;
    };

    // Utility for retrieving the first parameter in a variadic pack
    template<typename T, typename... Pack>
    struct FirstOf
    {
        using Type = T;
    };

    // FIXME: This needs to go away.
    // nvcc is broken and can't evaluate fold expressions on parameter packs without using a lambda 
    // This utility hides the ugly code away where it can't hurt anyone
    template<typename... Pack>
    struct SizeOfPack
    {
        static constexpr size_t F()
        {
            return ([&]() -> size_t { return sizeof(Pack); }() + ... + size_t(0));
        }

        enum : size_t
        {
            kValue = F()
        };
    };
}