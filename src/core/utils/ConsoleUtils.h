#pragma once

#include <cstdio>

//#define DISABLE_CONSOLE_COLOURS

namespace Flair
{
    enum ANSIColourCode : uint32_t
    {
        kANSIFgBlack = 30,
        kANSIFgRed = 31,
        kANSIFgGreen = 32,
        kANSIFgYellow = 33,
        kANSIFgBlue = 34,
        kANSIFgPurple = 35,
        kANSIFgTeal = 36,
        kANSIFgWhite = 37,
        kANSIFgDefault = 39,
        kANSIFgBrightBlack = 90,
        kANSIFgBrightRed = 91,
        kANSIFgBrightGreen = 92,
        kANSIFgBrightYellow = 93,
        kANSIFgBrightBlue = 94,
        kANSIFgBrightMagenta = 95,
        kANSIFgBrightCyan = 96,
        kANSIFgBrightWhite = 97,
        kANSIBgRed = 41,
        kANSIBgGreen = 42,
        kANSIBgYellow = 43,
        kANSIBgBlue = 44,
        kANSIBgPurple = 45,
        kANSIBgTeal = 46,
        kANSIBgWhite = 47,
        kANSIBgDefault = 49,
        kANSIBgBrightBlack = 100,
        kANSIBgBrightRed = 101,
        kANSIBgBrightGreen = 102,
        kANSIBgBrightYellow = 103,
        kANSIBgBrightBlue = 104,
        kANSIBgBrightMagenta = 105,
        kANSIBgBrightCyan = 106,
        kANSIBgBrightWhite = 107
    };

    class ANSIColourRAII
    {
    public:
#ifdef DISABLE_CONSOLE_COLOURS
        ANSIColourRAII(const uint32_t colour) { }
        ~ANSIColourRAII() { }
#else
        ANSIColourRAII(const uint32_t colour) { printf("\033[%um", colour); }
        ~ANSIColourRAII() { printf("\033[39m\033[49m"); }
#endif
    };

    template<typename... Pack>
    inline void printf_colour(const uint32_t colour, const char* fmt, Pack... pack)
    {
        ANSIColourRAII setColor(colour);
        printf(fmt, pack...);
    }

    // specialization for the case of not using any formatting (to silence "warning: format string is not a string literal")
    template<>
    inline void printf_colour(const uint32_t colour, const char* str)
    {
        ANSIColourRAII setColor(colour);
        printf("%s", str);
    }

    template<typename... Pack> inline void printf_red(const char* fmt, Pack... pack) { printf_colour(kANSIFgRed, fmt, pack...); }
    template<typename... Pack> inline void printf_yellow(const char* fmt, Pack... pack) { printf_colour(kANSIFgYellow, fmt, pack...); }
    template<typename... Pack> inline void printf_green(const char* fmt, Pack... pack) { printf_colour(kANSIFgGreen, fmt, pack...); }

    inline void NL(const int number = 1) 
    { 
        for (int i = 0; i < number; ++i) { printf("\n"); }
    }

}
