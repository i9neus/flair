#include "TestIterator.h"

namespace Flair
{

#ifdef FLAIR_ENABLE_MULTITHREADING
    #define IncAtomic(a) a++
    #define DecAtomic(a) a--
    #define LockGuard(a) std::lock_guard<std::mutex> mutexLock(a)
#else
    #define IncAtomic(a)
    #define DecAtomic(a)
    #define LockGuard(a)
#endif
    
    TestIterator::TestIterator(const Image3f& referenceImage, const int maxThreads) : 
        m_referenceImage(referenceImage),
        m_maxThreads(maxThreads)
    {
        m_referenceLum = m_referenceImage.ExtractLuminance();
    }

    void TestIterator::RunThread(const int startIdx, const int endIdx, const int numIters, Functor onInit, Functor onComplete)
    {
        IncAtomic(m_activeCount);
        
        Image3f decompressedImage(m_referenceImage);
        Codec codec;

        for (int iterIdx = startIdx; iterIdx < endIdx; ++iterIdx)
        {
            CompressedImageData compressedData;
            IterationData iterData(codec, iterIdx, numIters);

            try
            {
                {
                    LockGuard(m_mutex);
                    onInit(iterData);
                }

                // Encode and decode the image
                codec.Encode(m_referenceImage, compressedData);
                codec.Decode(compressedData, decompressedImage);
            }
            catch (const std::runtime_error& err)
            {
                std::printf("Encode/decode failed: %s\n", err.what());
            }
            
            const Image1f decompressedLum = decompressedImage.ExtractLuminance();
            iterData.stats = GenerateCodecStats(decompressedImage, m_referenceImage, compressedData);
            
            {
                LockGuard(m_mutex);
                onComplete(iterData);
            }

            IncAtomic(m_itersComplete);
        }

        DecAtomic(m_activeCount);
    }

    #ifdef FLAIR_ENABLE_MULTITHREADING
   
    // Run a series of compression/decompression cycles based on initialisation parameters supplied by onInit
    void TestIterator::Run(const int numIters, Functor onInit, Functor onComplete)
    {
        int numThreads = std::max(1, int(std::thread::hardware_concurrency()));
        if (m_maxThreads > 0) { numThreads = std::min(m_maxThreads, numThreads); }
        numThreads = std::min(numThreads, numIters);
        std::printf("Running on %i threads...\n", numThreads);
        
        m_itersComplete = 0;

        if (numThreads == 1)
        {
            RunThread(0, numIters, numIters, onInit, onComplete);
        }
        else
        {
            for (int threadIdx = 0; threadIdx < numThreads; ++threadIdx)
            {
                int startIdx = numIters * threadIdx / numThreads;
                int endIdx = numIters * (threadIdx + 1) / numThreads;
                m_workerThreads.emplace_back(&TestIterator::RunThread, this, startIdx, endIdx, numIters, onInit, onComplete);
                m_workerThreads.back().detach();
            }

            // Show a progress indicator while we wait for everything to finish
            do
            {
                std::printf("Running: %i of %i complete\r", m_itersComplete.load(), numIters);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            while (m_activeCount.load() > 0);

            std::printf("\n");
        }
    }

#else

    void TestIterator::Run(const int numIters, Functor onInit, Functor onComplete)
    {
        RunThread(0, numIters, numIters, onInit, onComplete);
    }

#endif
}