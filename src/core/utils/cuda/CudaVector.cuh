#pragma once

#include "CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace Cuda
    {        
        template<typename Type, int Alloc>
        class Vector
        {
        private:
            Type*               m_hostData;
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
                __host__ __forceinline__ Iterator& operator++() { ++m_idx; return *this; }
                __host__ __forceinline__ Iterator& operator--() { --m_m_idx; return *this; }
                __host__ __forceinline__ bool operator!=(const Iterator& other) const { return m_idx != other.m_idx; }
                __host__ __forceinline__ ItType& operator*() { return m_mem[m_idx]; }
                __host__ __forceinline__ ItType* operator->() { return &m_mem[idx]; }
            };

        private: 
            __forceinline__ __host__  void AssertHost() const
            {
                AssertMsg(Alloc == kCudaMemMirrored, "Vector does not have a host copy.");
            }

        public:           
            __host__ Vector() :
                m_hostData(nullptr),
                m_size(0),
                cu_deviceData(0)
            {
                static_assert(Alloc == kCudaMemDevice || Alloc == kCudaMemMirrored, "Alloc must be kCudaMemDevice or kCudaMemMirrored");
            }

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

            __host__ Vector(const Vector& other) = delete; 

            __host__ Vector(Vector&& other) : Vector()
            {
                this->operator=(std::move(other));
            }

            __host__ void Resize(const size_t newSize, const bool resync = true)
            {
                if (newSize == m_size) { return; }

                // Reallocate and move host data
                if (Alloc == kCudaMemMirrored)
                {
                    Type* newHostData = new Type[newSize];
                    if (m_hostData)
                    {
                        std::memcpy(newHostData, m_hostData, sizeof(Type) * newSize);
                        delete[] m_hostData;
                    }
                    m_hostData = newHostData;
                }

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
                if (resync) { Upload(); }
            }

            __host__ ~Vector()
            {
                if (cu_deviceData) { cudaFree(cu_deviceData); }
                if (m_hostData) { delete[] m_hostData; }
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }

            inline Type& operator[](const int idx) { AssertHost(); return m_hostData[idx]; }
            inline const Type& operator[](const int idx) const { AssertHost(); return m_hostData[idx]; }
            __host__ inline size_t Size() const { return m_size; }

            __host__ Vector& operator=(Vector&& other)
            {
                this->~Vector();

                cu_deviceData = other.cu_deviceData;
                m_size = other.m_size;
                m_hostData = other.m_hostData;
                other.cu_deviceData = nullptr;
                other.m_size = 0;
                other.m_hostData = nullptr;
                
                return *this;
            }

            __host__ Vector& operator=(const std::vector<Type>& otherCopy)
            {
                Resize(otherCopy.size(), false);

                if (m_size > 0)
                {
                    if (Alloc == kCudaMemMirrored)
                    {
                        memcpy(m_hostData, otherCopy.data(), sizeof(Type) * m_size);
                        IsOk(cudaMemcpy(cu_deviceData, m_hostData, sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                    }
                    else
                    {
                        IsOk(cudaMemcpy(cu_deviceData, otherCopy.data(), sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                    }
                }
                return *this;
            }

            __inline__ __host__ void Download()
            {
                if (Alloc == kCudaMemMirrored)
                {
                    IsOk(cudaMemcpy(m_hostData, cu_deviceData, sizeof(Type) * m_size, cudaMemcpyDeviceToHost));
                }
            }

            __inline__ __host__ void Upload()
            {
                if (Alloc == kCudaMemMirrored)
                {
                    IsOk(cudaMemcpy(cu_deviceData, m_hostData, sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                }
            }

            __host__ __forceinline__ Iterator<Type> begin() { AssertHost(); return Iterator<Type>(m_hostData, 0); }
            __host__ __forceinline__ Iterator<const Type> begin() const { AssertHost(); return Iterator<const Type>(m_hostData, 0); }
            __host__ __forceinline__ Iterator<Type> end() { AssertHost(); return Iterator<Type>(m_hostData, m_size); }
            __host__ __forceinline__ Iterator<const Type> end() const { AssertHost(); return Iterator<const Type>(m_hostData, m_size); }
        };

        template<typename Type, int Alloc>
        __host__ __inline__ void Swap(Vector<Type, Alloc>& a, Vector<Type, Alloc>& b)
        {
            Vector<Type, Alloc> temp = std::move(a);
            a = std::move(b);
            b = std::move(temp);
        }
    }
}
