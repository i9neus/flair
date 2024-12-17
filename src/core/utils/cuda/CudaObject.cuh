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
            Type            m_hostData;
            Type*           cu_deviceData;
            ComputeDevice   m_computeDevice;

        public:
            template<typename... Pack>
            __host__ Object(const ComputeDevice targetDevice, Pack... pack) :
                m_computeDevice(targetDevice)
            {                
                if (m_computeDevice == ComputeDevice::kCUDA)
                {
                    IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));
                }

                new (&m_hostData) Type(pack...);
                Upload();
            }

            __host__ ~Object()
            {
                cudaFree(cu_deviceData);
                m_hostData.~Type();
            }

            __host__ inline Type* operator->() { return &m_hostData; }
            __host__ inline const Type* operator->() const { return &m_hostData; }
            __host__ inline Type& operator*() { return m_hostData; }
            __host__ inline const Type& operator*() const { return m_hostData; }

            __host__ Type* GetComputeData() { return (m_computeDevice == ComputeDevice::kCUDA) ? cu_deviceData : &m_hostData; }
            __host__ const Type* GetComputeData() const { return (m_computeDevice == ComputeDevice::kCUDA) ? cu_deviceData : &m_hostData; }

            __host__ Object& operator=(const Type& hostCopy)
            {
                m_hostData = hostCopy;               
                return *this;
            }

            __inline__ __host__ Type& Download()
            {
                if (m_computeDevice == ComputeDevice::kCUDA)
                {
                    IsOk(cudaMemcpy(&m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
                }
                return m_hostData;
            }

            __inline__ __host__ void Upload()
            {
                if (m_computeDevice == ComputeDevice::kCUDA)
                {
                    IsOk(cudaMemcpy(cu_deviceData, &m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
                }
            }
        };   

        // Copy to host memory and upload to device
        template<typename Type>
        __host__ inline Object<Type>& operator<<=(Object<Type>& lhs, const Type& rhs)
        {
            lhs = rhs;
            lhs.Upload();
            return lhs;
        }

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
