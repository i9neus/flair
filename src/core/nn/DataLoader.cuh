#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace NN
    {
        class DataLoader
        {
        public:
            __host__ DataLoader() = default;

            __host__ virtual size_t Size() const = 0;
            __host__ virtual std::pair<float*, float*> operator[](const int idx) = 0;
            __host__ virtual std::pair<const float*, const float*> Data() const = 0;
        };
    }
}