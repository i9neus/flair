#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace Cuda
    {
        enum MirroredVectorFlags : uint32_t { kVectorExactMemory = 1 };
        
        template<typename Type>
        class MirroredVector
        {
        private:
            Type*               m_hostData;
            size_t              m_capacity;
            size_t              m_size;
            Type*               cu_deviceData;
            uint32_t            m_flags;

        public:           
            __host__ MirroredVector(const uint32_t flags = 0) :
                m_hostData(nullptr),
                m_capacity(0),
                m_size(0),
                cu_deviceData(0),
                m_flags(flags) {}

            __host__ MirroredVector(const size_t size) : MirroredVector(0)
            {
                Resize(size, false);
            }
            
            __host__ MirroredVector(const size_t size, Type initVal) : MirroredVector(0)
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

            __host__ ~MirroredVector()
            {
                if (cu_deviceData) { cudaFree(cu_deviceData); }
                if (m_hostData) { delete[] m_hostData; }
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }       

            inline Type& operator[](const int idx) { return m_data[idx]; }
            inline const Type& operator[](const int idx) const { return m_data[idx]; }
            __host__ inline size_t Size() const { return m_size; }

            __inline__ __host__ void Download()
            {
                IsOk(cudaMemcpy(&m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
            }

            __inline__ __host__ void Upload()
            {
                IsOk(cudaMemcpy(cu_deviceData, &m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
            }
        };
    }
}
