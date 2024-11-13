
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

#include "CudaUtils.cuh"

__global__ void Test()
{
    printf("%i\n", kKernelIdx);
}

void Run()
{
    IsOk(cudaSetDevice(0));

    Test << <1, 3 >> > ();
}

int main()
{
    try
    {
        Run();
    }
    catch (const std::runtime_error& err)
    {
        printf("Runtime error: %s\n", err.what());
    }
}

