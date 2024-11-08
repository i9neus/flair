#pragma once

#include <cmath>
#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <unordered_map>
#include <numeric>
#include <type_traits>
#include <cstring>
#include <memory>
#include "tinyformat.h"

#define FLAIR_ENABLE_MULTITHREADING

#ifdef __EMSCRIPTEN__
#undef FLAIR_ENABLE_MULTITHREADING
#endif

namespace Flair
{

#define Assert(condition) \
        if(!(condition)) {  \
            throw std::runtime_error(tfm::format("%s in %s (%d)", #condition, __FILE__, __LINE__)); \
        }

#define AssertMsg(condition, message) \
        if(!(condition)) {  \
            throw std::runtime_error(tfm::format("%s in %s (%d)", message, __FILE__, __LINE__)); \
        }

#define AssertFmt(condition, message, ...) \
        if(!(condition)) {  \
            char buffer[1024]; \
            std::snprintf(buffer, 1024, message, __VA_ARGS__); \
            throw std::runtime_error(tfm::format("%s in %s (%d)", buffer, __FILE__, __LINE__)); \
        }

    using MagicType = uint32_t;
}