#pragma once

#include "CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include <vector>

namespace Flair
{  
    namespace Cuda
    {                
        template<typename T>
        __global__ static void ConstantInitialiseVectorKernel(T* data, const size_t dataSize, const T value)
        {
            if (kKernelIdx < dataSize) { data[kKernelIdx] = value; }
        }

        template<typename Type>
        class Vector
        {
            template<typename OtherType> friend std::vector<OtherType>& operator<<=(std::vector<OtherType>&, const Vector<OtherType>&);

        private:
            size_t              m_size = 0;
            Type*               cu_deviceData = nullptr;
            std::vector<Type>   m_hostData;
            ComputeDevice       m_computeDevice = ComputeDevice::kCUDA;

        public:
            __host__ Vector(const ComputeDevice targetDevice, const size_t size = 0) :
                m_size(0),
                cu_deviceData(0),
                m_computeDevice(targetDevice)
            {
                Resize(size);
            }

            __host__ Vector(const ComputeDevice targetDevice, const size_t size, const Type& initVal) : Vector(targetDevice, size)
            {
                Fill(initVal);
            }

            __host__ Vector(const Vector&) = delete;
            __host__ Vector& operator=(const Vector& other) = delete;
            __host__ Vector(Vector&& other) { *this = std::move(other); }

            __host__ ~Vector()
            {
                if (cu_deviceData)
                {
                    cudaFree(cu_deviceData);
                    cu_deviceData = nullptr;
                }
            }

            __host__ void Resize(const size_t newSize)
            {
                if (m_computeDevice == ComputeDevice::kCUDA)
                {
                    if (newSize == m_size) { return; }

                    if (newSize == 0)
                    {
                        cudaFree(cu_deviceData);
                        cu_deviceData = nullptr;
                        m_size = 0;
                    }
                    else
                    {
                        // Reallocate and move device data
                        Type* newDeviceData;
                        IsOk(cudaMalloc((void**)&newDeviceData, sizeof(Type) * newSize));
                        Assert(newDeviceData);
                        if (cu_deviceData)
                        {
                            IsOk(cudaMemcpy(newDeviceData, cu_deviceData, sizeof(Type) * std::min(newSize, m_size), cudaMemcpyDeviceToDevice));
                            cudaFree(cu_deviceData);
                        }

                        m_size = newSize;
                        cu_deviceData = newDeviceData;
                    }
                }
                else
                {
                    m_hostData.resize(newSize);
                    m_size = newSize;
                }
            }

            // Fills the vector with a constant value
            __host__ inline void Fill(const Type& value)
            {
                if (m_computeDevice == ComputeDevice::kCUDA)
                {
                    if (m_size > 0)
                    {
                        const int kNumBlocks = (m_size + 255) / 256;
                        ConstantInitialiseVectorKernel <<< kNumBlocks, 256 >> > (cu_deviceData, m_size, value);
                    }
                }
                else
                {
                    for (auto& f : m_hostData) { f = value; }
                }
            }

            __host__ Vector& operator<<=(const std::vector<Type>& rhs)
            {
                Resize(rhs.size());
                if (m_computeDevice == ComputeDevice::kCUDA)
                {
                    if (m_size > 0)
                    {
                        IsOk(cudaMemcpy(cu_deviceData, rhs.data(), sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                    }
                }
                else
                {
                    m_hostData = rhs;
                }
                return *this;
            }

            __host__ Vector& operator=(Vector&& other)
            {
                this->~Vector();
                cu_deviceData = other.cu_deviceData;
                m_hostData = std::move(other.m_hostData);
                m_size = other.m_size;
                m_computeDevice = other.m_computeDevice;

                other.cu_deviceData = nullptr;
                other.m_size = 0;                

                return *this;
            }

            __host__ Type* GetComputeData() { return (m_computeDevice == ComputeDevice::kCUDA) ? cu_deviceData : m_hostData.data(); }
            __host__ const Type* GetComputeData() const { return (m_computeDevice == ComputeDevice::kCUDA) ? cu_deviceData : m_hostData.data(); }

            __host__ Vector& To(const ComputeDevice targetDevice)
            {
                if (m_size == 0 || m_computeDevice == targetDevice)
                {
                    return *this;
                }
                else
                {
                    Vector<Type> temp(targetDevice, m_size);
                    if (targetDevice == ComputeDevice::kCUDA) { temp <<= m_hostData; }
                    else { temp.m_hostData <<= *this; }
                    *this = std::move(temp);
                }

                return *this;
            }

            __host__ inline Type& operator[](const size_t idx)             
            {
                AssertFmt(m_computeDevice == ComputeDevice::kCPU, "Cannot index data on CUDA device from host. Copy data to CPU with To() first.");
                AssertDebugFmt(idx < m_size, "Array index %i out of bounds [0, %i)", idx, m_size);
                
                return m_hostData[idx];
            }
            __host__ inline const Type& operator[](const size_t idx) const { return const_cast<Vector&>(*this)[idx]; }

            __host__ inline size_t Size() const { return m_size; }
            __host__ inline size_t IsEmpty() const { return m_size == 0; }
            __host__ ComputeDevice GetComputeDevice() const { return m_computeDevice; }
            __host__ bool IsCUDA() const { return m_computeDevice == ComputeDevice::kCUDA; }
            __host__ bool IsCPU() const { return m_computeDevice == ComputeDevice::kCPU; }
        };

        // Download from device and copy to host memory
        template<typename Type>
        __host__ static std::vector<Type>& operator<<=(std::vector<Type>& lhs, const Vector<Type>& rhs)
        {
            if (rhs.GetComputeDevice() == ComputeDevice::kCUDA)
            {
                if (!rhs.IsEmpty())
                {
                    lhs.resize(rhs.Size());
                    IsOk(cudaMemcpy(lhs.data(), rhs.cu_deviceData, sizeof(Type) * rhs.Size(), cudaMemcpyDeviceToHost));
                }
            }
            else
            {
                lhs = rhs.m_hostData;
            }
            return lhs;
        }        

        template<typename Type>
        __host__ __inline__ void Swap(Vector<Type>& a, Vector<Type>& b)
        {
            Vector<Type> temp = std::move(a);
            a = std::move(b);
            b = std::move(temp);
        }       
    }
}
