#pragma once

#include <chrono>

namespace Flair
{
    class HighResTimer
    {
    public:
        HighResTimer() : m_startTime(std::chrono::high_resolution_clock::now()) {}

        inline float Get() const { return float(std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_startTime).count()); }
        inline void Reset() { m_startTime = std::chrono::high_resolution_clock::now(); }

        inline float operator()() const { return Get(); }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
    };
}
