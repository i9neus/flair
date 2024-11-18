#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    namespace Cuda
    {
        /*template<typename Type>
        class DeviceObject
        {
        private:
            Type* cu_data;

        public:
            __host__ DeviceObject()
            {
                IsOk(cudaMalloc((void**)&cu_data, sizeof(Type)));
                Assert(cu_data);
                printf_green("Allocated %i bytes of device data.\n", sizeof(Type));
            }

            __host__ ~DeviceObject()
            {
                cudaFree(cu_data);
                printf_green("Released %i bytes of device data.\n", sizeof(Type));
            }

            __host__ Type* GetDeviceData() { return cu_data; }

            __host__ DeviceObject& operator=(const Type& hostCopy)
            {
                IsOk(cudaMemcpy(cu_data, &hostCopy, sizeof(Type), cudaMemcpyHostToDevice));
                return *this;
            }

            __host__ void Get(Type& hostCopy)
            {
                IsOk(cudaMemcpy(&hostCopy, cu_data, sizeof(Type), cudaMemcpyDeviceToHost));
            }
        };*/

        template<typename Type>
        class HostDeviceObject
        {
        private:
            Type  m_hostData;
            Type* cu_deviceData;

        public:
            template<typename... Pack>
            __host__ HostDeviceObject(Pack... pack) : 
                HostDeviceObject()
            {
                m_hostData = Type(pack...);
                Upload();
            }

            __host__ HostDeviceObject()
            {
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));
                Assert(cu_deviceData);
                //printf_green("Allocated %i bytes of device data.\n", sizeof(Type));
            }

            __host__ ~HostDeviceObject()
            {
                cudaFree(cu_deviceData);
                //printf_green("Released %i bytes of device data.\n", sizeof(Type));
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }

            __host__ inline Type* operator->() { return &m_hostData; }
            __host__ inline const Type* operator->() const { return &m_hostData; }
            __host__ inline Type& operator*() { return m_hostData; }
            __host__ inline const Type& operator*() const { return m_hostData; }

            __host__ HostDeviceObject& operator=(const Type& hostCopy)
            {
                m_hostData = hostCopy;
                Upload();
                return *this;
            }

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
