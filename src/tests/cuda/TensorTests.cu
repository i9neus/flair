#include "TensorTests.cuh"

#include "core/utils/cuda/Tensor1D.cuh"
#include "core/utils/cuda/Tensor2D.cuh"
#include "core/utils/cuda/ManagedObject.cuh"

namespace Flair
{
    template<bool Transpose>
    __global__  void KernelMul(const Tensor2D<4, 4>* M, const Tensor1D<4>* v, Tensor1D<4>* r)
    {
        __shared__ float scratch[4][4];

        *r = *v;
        Mul<Transpose>(*M, *r, scratch);
    }

    __host__ void RunTensorTests()
    {
        constexpr float kErrorThreshold = 1e-6;
        
        Cuda::HostDeviceObject<Tensor2D<4, 4>> M;
        Cuda::HostDeviceObject<Tensor1D<4>> v;
        Cuda::HostDeviceObject<Tensor1D<4>> r;

        // NOTE: Column-major order 
        M = Tensor2D<4, 4>({ {0.652467807974029, 0.633070356251368, 0.68281308686666, 0.566351831093323},
                               {0.935202196659332, 0.976187756902101, 0.238451694824191, 0.637562295790242},
                               {0.101098380420291, 0.64552469382196, 0.159522225810158, 0.813787851275935},
                               {0.904785470451441, 0.640712457447006, 0.306539727511699, 0.756197597784415} });

        // Input vector
        v = Tensor1D<4>({ 0.876688377753995, 0.019128400638492, 0.542616681289372, 0.352370872609403 });        

        // Targets for multiply and multiply transpose
        const Tensor1D<4> targetMulT({ {1.15419222757901}, {1.19260005697745}, {0.474294186123722}, {1.23826628790775} });
        const Tensor1D<4> targetMul( {{0.963577579819826}, {1.1497192089129}, {0.797750589019602}, {1.21674648559447}});

        // Test and check errors
        KernelMul<false> << <1, 16 >> >(M.GetDeviceData(), v.GetDeviceData(), r.GetDeviceData());
        r.Download();
        const float errorMul = CwiseMax(Abs(*r - targetMul));
        AssertFmt(errorMul < kErrorThreshold, "TEST FAILED: Tensor2D mul: %f", errorMul);

        KernelMul<true> << <1, 16 >> > (M.GetDeviceData(), v.GetDeviceData(), r.GetDeviceData());
        r.Download();
        const float errorMulT = CwiseMax(Abs(*r - targetMulT));
        AssertFmt(errorMulT < kErrorThreshold, "TEST FAILED: Tensor2D mul transpose: %f", errorMulT);
        
    }
}