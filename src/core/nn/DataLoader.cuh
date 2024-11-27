#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace NN
    {
        template<typename SampleT>
        class DataLoader
        {
        public:
            using Sample = SampleT;

            __host__ DataLoader() = default;

            __host__ virtual size_t Size() const = 0;
            __host__ virtual std::pair<const std::vector<Sample>*, const std::vector<Sample>*> Data() const = 0;
        };
    }
}