#pragma once

#include "CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include <vector>

namespace Flair
{  
    namespace Cuda
    {                
        //template<typename OtherType> std::vector<OtherType>& operator<<=(std::vector<OtherType>&, Vector<OtherType>&);

        template<typename T>
        __global__ static void ConstantInitialiseVectorKernel(T* data, const size_t dataSize, const T value)
        {
            if (kKernelIdx < dataSize) { data[kKernelIdx] = value; }
        }

        template<typename Type>
        class Vector
        {
            template<typename OtherType> friend std::vector<OtherType>& operator<<=(std::vector<OtherType>&, Vector<OtherType>&);

        private:
            size_t              m_size = 0;
            Type*               cu_deviceData = nullptr;

        public:
            __host__ Vector() :
                m_size(0),
                cu_deviceData(0) {}

            __host__ Vector(const size_t size) : Vector()
            {
                Resize(size);
            }

            __host__ Vector(const size_t size, Type initVal) : Vector()
            {
                Resize(size);
                Fill(initVal);
            }

            __host__ Vector(const Vector&) = delete;
            
            __host__ Vector(Vector&& other)
            {
                *this = std::move(other);
            }

            __host__ ~Vector()
            {
                if (cu_deviceData) 
                { 
                    cudaFree(cu_deviceData); 
                }
            }

            __host__ void Resize(const size_t newSize)
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

            // Fills the vector with a constant value
            __host__ void Fill(const Type& value)
            {
                if (m_size > 0)
                {
                    const int kNumBlocks = (m_size + 255) / 256;
                    ConstantInitialiseVectorKernel << <kNumBlocks, 256 >> > (cu_deviceData, m_size, value);
                }
            }

            __host__ Vector& operator<<=(const std::vector<Type>& rhs)
            {
                Resize(rhs.size());
                if (m_size > 0)
                {
                    IsOk(cudaMemcpy(cu_deviceData, rhs.data(), sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                }
                return *this;
            }

            __host__ Vector& operator=(const Vector& other) = delete;

            __host__ Vector& operator=(Vector&& other)
            {
                this->~Vector();
                cu_deviceData = other.cu_deviceData;
                m_size = other.m_size;
                other.cu_deviceData = nullptr;
                other.m_size = 0;

                return *this;
            }

            //__host__ __inline__ Vector& operator<<=(const std::vector<Type>& rhs) { *this = rhs; }

            __host__ Type* GetDeviceData() { return cu_deviceData; }
            __host__ const Type* GetDeviceData() const { return cu_deviceData; }

            __host__ inline size_t Size() const { return m_size; }
            __host__ inline size_t IsEmpty() const { return m_size == 0; }
        };

        template<typename Type>
        __host__ __inline__ void Swap(Vector<Type>& a, Vector<Type>& b)
        {
            Vector<Type> temp = std::move(a);
            a = std::move(b);
            b = std::move(temp);
        }

        // Download from device and copy to host memory
        template<typename Type>
        __host__ static std::vector<Type>& operator<<=(std::vector<Type>& lhs, Vector<Type>& rhs)
        {
            lhs.resize(rhs.Size());
            if (!lhs.empty())
            {
                IsOk(cudaMemcpy(lhs.data(), rhs.cu_deviceData, sizeof(Type) * rhs.Size(), cudaMemcpyDeviceToHost));
            }
            return lhs;
        }

        /*template<typename Type, int Alloc>
        class Vector
        {
            template<typename OtherType, int OtherAlloc> friend std::vector<OtherType>& operator<<=(std::vector<OtherType>&, Vector<OtherType, OtherAlloc>&);

        private:
            std::vector<Type>   m_hostData;
            size_t              m_size;
            Type*               cu_deviceData;

        private: 
            __forceinline__ __host__  void AssertHost() const
            {
                AssertMsg(Alloc == kCudaMemMirrored, "Vector does not have a host copy.");
            }

        public:           
            __host__ Vector() :
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
                Fill(initVal);
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
                    m_hostData.resize(newSize);
                    m_hostData.shrink_to_fit();                    
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

                // Update and resyncb
                m_size = newSize;
                if (Alloc == kCudaMemMirrored && resync) 
                { 
                    Upload(); 
                }
            }

            __host__ ~Vector()
            {
                if (cu_deviceData) { cudaFree(cu_deviceData); }
            }

            __host__ Type* GetDeviceData() { return cu_deviceData; }
            __host__ const Type* GetDeviceData() const { return cu_deviceData; }

            inline Type& operator[](const int idx) { AssertHost(); return m_hostData[idx]; }
            inline const Type& operator[](const int idx) const { AssertHost(); return m_hostData[idx]; }
            __host__ inline size_t Size() const { return m_size; }
            __host__ inline size_t IsEmpty() const { return m_size == 0; }

            __host__ Vector& operator=(Vector&& other)
            {
                this->~Vector();

                cu_deviceData = other.cu_deviceData;
                m_size = other.m_size;
                m_hostData = std::move(other.m_hostData);
                other.cu_deviceData = nullptr;
                other.m_size = 0;
                other.m_hostData = std::vector<Type>();
                
                return *this;
            }

            // Copy to host memory and upload to device
            __host__ Vector& operator<<=(const std::vector<Type>& rhs)
            {
                Resize(rhs.size(), false);
                if (m_size > 0)
                {
                    if (Alloc == kCudaMemMirrored)
                    {
                        memcpy(m_hostData.data(), rhs.data(), sizeof(Type) * m_size);
                    }
                    IsOk(cudaMemcpy(cu_deviceData, rhs.data(), sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                }
                return *this;
            }

            // Fills the vector with a constant value
            __host__ void Fill(const Type& value)
            { 
                if (m_size > 0)
                {
                    if (Alloc == kCudaMemMirrored)
                    {
                        for (auto& f : m_hostData) { f = value; }
                    }
                    else
                    {
                        const int kNumBlocks = (m_size + 255) / 256;
                        ConstantInitialiseVectorKernel << <kNumBlocks, 256 >> > (cu_deviceData, m_size, value);
                    }
                }
            }

            __host__ Vector& operator=(const std::vector<Type>& otherCopy)
            {
                Resize(otherCopy.size(), false);
                if (m_size > 0)
                {
                    AssertMsg(Alloc == kCudaMemMirrored, "Operator = for device-only vectors does nothing. Use <<= instead.");
                    memcpy(m_hostData.data(), otherCopy.data(), sizeof(Type) * m_size);
                }
                return *this;
            }

            __inline__ __host__ std::vector<Type>& Download()
            {
                AssertHost();
                IsOk(cudaMemcpy(m_hostData.data(), cu_deviceData, sizeof(Type) * m_size, cudaMemcpyDeviceToHost));
                return m_hostData;
            }

            __inline__ __host__ void Upload()
            {
                if (Alloc == kCudaMemMirrored)
                {
                    IsOk(cudaMemcpy(cu_deviceData, m_hostData.data(), sizeof(Type) * m_size, cudaMemcpyHostToDevice));
                }
            }

            __host__ __forceinline__ std::vector<Type>::iterator begin() { AssertHost(); return m_hostData.begin(); }
            __host__ __forceinline__ std::vector<Type>::const_iterator begin() const { AssertHost(); return m_hostData.cbegin(); }
            __host__ __forceinline__ std::vector<Type>::iterator end() { AssertHost(); return m_hostData.end(); }
            __host__ __forceinline__ std::vector<Type>::const_iterator end() const { AssertHost(); return m_hostData.cend(); }

            __host__ __forceinline__ std::vector<Type>& Data() { AssertHost(); return m_hostData; }
            __host__ __forceinline__ const std::vector<Type>& Data() const { AssertHost(); return m_hostData; }
        };

        template<typename Type, int Alloc>
        __host__ __inline__ void Swap(Vector<Type, Alloc>& a, Vector<Type, Alloc>& b)
        {
            Vector<Type, Alloc> temp = std::move(a);
            a = std::move(b);
            b = std::move(temp);
        }  

        // Download from device and copy to host memory
        template<typename Type, int Alloc>
        __host__ static std::vector<Type> DownloadVec(Cuda::Vector<Type, Alloc>& rhs)
        {
            std::vector<Type> lhs;
            return lhs;
        }

        // Download from device and copy to host memory
        template<typename Type, int Alloc>
        __host__ static std::vector<Type>& operator<<=(std::vector<Type>& lhs, Vector<Type, Alloc>& rhs)
        {
            lhs.resize(rhs.Size());
            if (!lhs.empty())
            {
                if (Alloc == kCudaMemMirrored)
                {
                    rhs.Download();
                    memcpy(lhs.data(), rhs.m_hostData.data(), sizeof(Type) * rhs.Size());
                }
                else
                {
                    IsOk(cudaMemcpy(lhs.data(), rhs.cu_deviceData, sizeof(Type) * rhs.Size(), cudaMemcpyDeviceToHost));
                }
            }
            return lhs;
        }*/
    }
}
