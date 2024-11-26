#pragma once

#include "CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace Cuda
    {        
        template<typename Type>
        class Vector
        {
        private:
            Type*               m_hostData;
            size_t              m_capacity;
            size_t              m_size;
            Type*               cu_deviceData;

        public:
            template<typename ItType>
            class Iterator
            {
                friend class Vector;
                Type* m_mem;
                size_t m_idx;

            private:
                Iterator(Type* mem, const int idx) : m_mem(mem), m_idx(idx) {}

            public:
                __device__ __forceinline__ Iterator& operator++() { ++m_idx; return *this; }
                __device__ __forceinline__ Iterator& operator--() { --m_m_idx; return *this; }
                __device__ __forceinline__ bool operator!=(const Iterator& other) const { return m_idx != other.m_idx; }
                __device__ __forceinline__ ItType& operator*() { return m_mem[m_idx]; }
                __device__ __forceinline__ ItType* operator->() { return &m_mem[idx]; }
            };

        public:           
            __host__ Vector() :
                m_hostData(nullptr),
                m_capacity(0),
                m_size(0),
                cu_deviceData(0){}

            __host__ Vector(const size_t size) : Vector()
            {
                Resize(size, true);
            }
            
            __host__ Vector(const size_t size, Type initVal) : Vector(0)
            {
                Resize(size, false);
                for (int i = 0; i < m_size; ++i) { m_hostData[i] = initVal; }
                Upload();
            }

            __host__ void Resize(const size_t newSize, const bool resync = true)
            {
                if (newSize == m_size) { return; }

                // Reallocate and move host data
                Type* newHostData = new Type[newSize];
                if (m_hostData)
                {
                    std::memcpy(newHostData, m_hostData, sizeof(Type) * newSize);
                    delete[] m_hostData;
                }
                m_hostData = newHostData;

                // Reallocate and move device data
                Type* newDeviceData;
                IsOk(cudaMalloc((void**)&newDeviceData, sizeof(Type) * newSize));
                Assert(newDeviceData);
                if (cu_deviceData)
                {
                    cudaFree(cu_deviceData);
                }
                cu_deviceData = newDeviceData;

                // Update and resync
                m_size = newSize;
                if (resync)
                {
                    Upload();
                }
            }

            __host__ ~Vector()
            {
                if (cu_deviceData) { cudaFree(cu_deviceData); }
                if (m_hostData) { delete[] m_hostData; }
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }       

            inline Type& operator[](const int idx) { return m_hostData[idx]; }
            inline const Type& operator[](const int idx) const { return m_hostData[idx]; }
            __host__ inline size_t Size() const { return m_size; }

            __inline__ __host__ void Download()
            {
                IsOk(cudaMemcpy(&m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
            }

            __inline__ __host__ void Upload()
            {
                IsOk(cudaMemcpy(cu_deviceData, &m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
            }

            __host__ __forceinline__ Iterator<Type> begin() { return Iterator<Type>(m_hostData, 0); }
            __host__ __forceinline__ Iterator<const Type> begin() const { return Iterator<const Type>(m_hostData, 0); }
            __host__ __forceinline__ Iterator<Type> end() { return Iterator<Type>(m_hostData, m_size); }
            __host__ __forceinline__ Iterator<const Type> end() const { return Iterator<const Type>(m_hostData, m_size); }
        };
    }
}
