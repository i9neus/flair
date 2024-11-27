#pragma once

#include "CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace Cuda
    {
        template<typename Type, int Alloc>
        class Object
        {
        private:
            Type* m_hostData;
            Type* cu_deviceData;

        private:
            __forceinline__ __host__  void AssertHost() const
            {
                AssertMsg(Alloc == kCudaMemMirrored, "Vector does not have a host copy.");
            }

        public:
            template<typename... Pack>
            __host__ Object(Pack... pack)
            {
                static_assert(Alloc == kCudaMemDevice || Alloc == kCudaMemMirrored, "Alloc must be kCudaMemDevice or kCudaMemMirrored");
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));

                if (Alloc == kCudaMemMirrored)
                {
                    *m_hostData = new Type(pack...);
                    Upload();
                }
            }

            __host__ Object()
                : m_hostData(nullptr), cu_deviceData(nullptr)
            {
                static_assert(Alloc == kCudaMemDevice || Alloc == kCudaMemMirrored, "Alloc must be kCudaMemDevice or kCudaMemMirrored");
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));

                if (Alloc == kCudaMemMirrored)
                {
                    m_hostData = new Type();
                    Upload();
                }
            }

            __host__ ~Object()
            {
                cudaFree(cu_deviceData);
                if (m_hostData) { delete m_hostData; }
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }

            __host__ inline Type* operator->() { return m_hostData; }
            __host__ inline const Type* operator->() const { return m_hostData; }
            __host__ inline Type& operator*() { return *m_hostData; }
            __host__ inline const Type& operator*() const { return *m_hostData; }

            __host__ Object& operator=(const Type& hostCopy)
            {
                if (Alloc == kCudaMemMirrored)
                {
                    *m_hostData = hostCopy;
                    IsOk(cudaMemcpy(cu_deviceData, m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
                }
                else
                {
                    IsOk(cudaMemcpy(cu_deviceData, &hostCopy, sizeof(Type), cudaMemcpyHostToDevice));
                }
                return *this;
            }

            __inline__ __host__ void Download()
            {
                AssertHost();
                IsOk(cudaMemcpy(m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
            }

            __inline__ __host__ void Upload()
            {
                AssertHost();
                IsOk(cudaMemcpy(cu_deviceData, m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
            }
        };
    }
}
