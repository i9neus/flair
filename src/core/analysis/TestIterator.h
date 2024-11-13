#pragma once

#include "../Includes.h"
#include "../image/Image.h"
#include "../Codec.h"
#include "Metrics.h"

#include <functional>

#ifdef FLAIR_ENABLE_MULTITHREADING
#include <thread>
#include <atomic>
#include <mutex>
#endif

namespace Flair
{
    class TestIterator  
    {
    public:
         struct IterationData
         {
             IterationData(Codec& c, const int _i, const int _N) : codec(c), i(_i), N(_N) {}

             Codec& codec;
             CodecStats stats;
             const int i;
             const int N;
         };

        using Functor = std::function<void(IterationData&)>;

    private:
        const Image3f&              m_referenceImage;
        Image1f                     m_referenceLum;

#ifdef FLAIR_ENABLE_MULTITHREADING
        std::vector<std::thread>    m_workerThreads;
        std::mutex                  m_mutex;
        std::atomic<int>            m_activeCount;
        std::atomic<int>            m_itersComplete;
#endif
        int                         m_maxThreads;

    public:
        TestIterator(const Image3f& inputImage, int maxThreads = -1);

        void Run(const int numIters, Functor onUpdateParam, Functor onComplete);

    private:
        void RunThread(const int startIdx, const int endIdx, const int numIters, Functor onInit, Functor onComplete);

    };

}