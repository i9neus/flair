#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "thirdparty/tinyformat/tinyformat.h"
#include "core/math/MathUtils.h"

namespace Flair
{
    template<typename Type>
    class TensorIterator
    {
        Type* m_data;
        int     m_idx;

    public:
        __host__ __device__ TensorIterator(Type* mem, const int idx) : m_data(mem), m_idx(idx) {}

        __host__ __device__ __forceinline__ TensorIterator& operator++() { ++m_idx; return *this; }
        __host__ __device__ __forceinline__ bool operator!=(const TensorIterator& other) const { return m_idx != other.m_idx; }
        __host__ __device__ __forceinline__ Type& operator*() { return m_data[m_idx]; }
        __host__ __device__ __forceinline__ Type* operator->() { return &m_data[m_idx]; }
    };
    
}