#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "thirdparty/tinyformat/tinyformat.h"

namespace Flair
{
    // Sequentially sums the contents of a list using the supplied functor
    template<int N, typename Lambda>
    __host__ __device__ __inline__ auto SequentialSum(Lambda lambda)
    {
        using Type = decltype(lambda(0));
        Type sum = 0;
        for (int n = 0; n < N; ++n)
        {
            sum += lambda(n);
        }

        return sum;
    }

    // Map-reduces the contents of a list using the supplied functor. Potentially avoids loss of precision
    // due to rounding errors as sum grows large
    template<int N, typename Lambda>
    __host__ __device__ __inline__ auto MapReduceSum(Lambda lambda)
    {
        // Map
        using Type = decltype(lambda(0));
        Type scratch[(N + 1) / 2];
        for (int n = 0; n < N; n += 2)
        {
            scratch[n >> 1] = lambda(n);
            if (n + 1 < N) { scratch[n >> 1] += lambda(n + 1); }
        }

        // Reduce
        constexpr int Shift = ((N & (N - 1)) == 0) ? 0 : 1;
        for (int reduceMask = 4; (reduceMask >> Shift) <= N; reduceMask <<= 1)
        {
            for (int n = 0; n + (reduceMask >> 1) < N; n += reduceMask)
            {
                scratch[n >> 1] += scratch[(n + (reduceMask >> 1)) >> 1];
            }
        }

        return scratch[0];
    }
}