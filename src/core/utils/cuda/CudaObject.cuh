#pragma once

#include "CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace Cuda
    {
        template<typename Type>
        class Object
        {
        private:
            Type m_hostData;
            Type* cu_deviceData;

        public:
            template<typename... Pack>
            __host__ Object(Pack... pack)
            {
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));

                new (&m_hostData) Type(pack...);
                Upload();
            }

            __host__ Object()
            {
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));

                new (&m_hostData) Type;
                Upload();
            }

            __host__ ~Object()
            {
                cudaFree(cu_deviceData);
                m_hostData.~Type();
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }
            __host__ const Type* GetDeviceData() const { return cu_deviceData; }

            __host__ inline Type* operator->() { return &m_hostData; }
            __host__ inline const Type* operator->() const { return &m_hostData; }
            __host__ inline Type& operator*() { return m_hostData; }
            __host__ inline const Type& operator*() const { return m_hostData; }

            __host__ Object& operator=(const Type& hostCopy)
            {
                m_hostData = hostCopy;               
                return *this;
            }

            // Copy to host memory and upload to device
            __host__ Object& operator<<=(const Type hostCopy)
            {
                m_hostData = hostCopy;
                Upload();
                return *this;
            }

            __inline__ __host__ Type& Download()
            {
                IsOk(cudaMemcpy(&m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
                return m_hostData;
            }

            __inline__ __host__ void Upload()
            {
                IsOk(cudaMemcpy(cu_deviceData, &m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
            }
        };   

        // Download from device and copy to host memory
        template<typename Type>
        __host__ inline Type& operator<<=(Type& lhs, Object<Type>& rhs)
        {
            rhs.Download();
            lhs = *rhs;
            return lhs;
        }
    }   
}
