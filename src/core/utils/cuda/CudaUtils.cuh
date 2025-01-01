#pragma once

#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#include "core/Includes.h"
#include <type_traits>
#include <math.h>

#define kKernelIdx				(blockIdx.x * blockDim.x + threadIdx.x)	
#define kThreadIdx              threadIdx.x
#define kBlockIdx               blockIdx.x
#define kBlockDim               blockDim.x
#define kKernelDim              (blockDim.x * gridDim.x)
#define kWarpLane				(threadIdx.x & 31)

//template<typename T> __device__ __forceinline__ T kKernelPos() { return T(typename T::kType(kKernelX), typename T::kType(kKernelY)); }
//template<typename T> __device__ __forceinline__ T kKernelDims() { return T(typename T::kType(kKernelWidth), typename T::kType(kKernelHeight)); }

#if defined(_DEBUG)
#define IsCudaDebug() true
#else
#define IsCudaDebug() false
#endif

#if defined(__CUDA_ARCH__)

    // CUDA device-side asserts. We don't use assert() here because it's optimised out in the release build.
#define CudaAssert(condition) \
        if(!(condition)) {  \
            printf("Device assert: %s in %s (%d)\n", #condition, __FILE__, __LINE__); \
            asm("trap;"); \
        }

#define CudaAssertFmtImpl(condition, message, ...) \
        if(!(condition)) {  \
            printf(message, __VA_ARGS__); \
            asm("trap;"); \
        }
#define CudaAssertFmt(condition, message, ...) CudaAssertFmtImpl(condition, message"\n", __VA_ARGS__)

#if defined(_DEBUG)
#define CudaAssertDebug(condition) CudaAssert(condition)
#define CudaAssertDebugMsg(condition, message) CudaAssertFmt(condition, message)
#define CudaAssertDebugFmt(condition, message, ...) CudaAssertFmt(condition, message, __VA_ARGS__)
#else
#define CudaAssertDebug(condition)
#define CudaAssertDebugMsg(condition, message)
#define CudaAssertDebugFmt(condition, message, ...)
#endif

#else // __CUDA_ARCH__

#define CudaAssert(condition) Assert(condition)
#define CudaAssertFmt(condition, message, ...)  AssertFmt(condition, message, __VA_ARGS__)

#if defined(_DEBUG)
#define CudaAssertDebug(condition) Assert(condition)
#define CudaAssertDebugFmt(condition, message, ...) AssertFmt(condition, message, __VA_ARGS__)
#else
#define CudaAssertDebug(condition)
#define CudaAssertDebugFmt(condition, message, ...)
#endif

#endif // __CUDA_ARCH__

template <typename T>
__host__ inline void CudaHostAssert(T result, char const* const func, const char* const file, const int line)
{
    if (result != 0)
    {
        AssertFmt(false,
            "(file %s, line %d) CUDA returned error code=%d(%s) \"%s\" \n",
            file, line, (unsigned int)result, _cudaGetErrorEnum(result), func);
    }
}

#define IsOk(val) CudaHostAssert((val), #val, __FILE__, __LINE__)

#define CudaPrintVar(var, kind) printf(#var ": %" #kind "\n", var)

// Defines a generic kernel function that invokes the method in the referenced class
#define DEFINE_KERNEL_PASSTHROUGH_ARGS(FunctionName) \
        template<typename ObjectType, typename... Pack>\
        __global__ void Kernel##FunctionName(ObjectType* object, Pack... pack) \
        { \
            CudaAssert(object); \
            object->FunctionName(pack...); \
        }

#define DEFINE_KERNEL_PASSTHROUGH(FunctionName) \
        template<typename ObjectType>\
        __global__ void Kernel##FunctionName(ObjectType* object) \
        { \
            CudaAssert(object); \
            object->FunctionName(); \
        }

enum ContainerFlags : int { kCudaMemDevice = 1, kCudaMemMirrored = 2 };

enum class ComputeDevice { kCUDA, kCPU };