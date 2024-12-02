#include "TensorTests.cuh"

#include "core/nn/Tensor1D.cuh"
#include "core/nn/Tensor2D.cuh"
#include "core/nn/TensorOps.cuh"
#include "core/utils/cuda/CudaObject.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    template<bool Transpose, int N, bool HasGrad>
    __global__  void KernelMulSquare(const Tensor2D<N, N, HasGrad>* X, const Tensor1D<N, HasGrad>* v, Tensor1D<N, HasGrad>* w)
    {
        __shared__ float scratch[N][N];

        MulImpl<Transpose>(*X, *v, *w, scratch);
    }

    
    template<bool Transpose, int N, int M, int V, int W, bool HasGrad>
    __global__  void KernelMulNonSquare(const Tensor2D<N, M, HasGrad>* X, const Tensor1D<V, HasGrad>* v, Tensor1D<W, HasGrad>* w)
    {
        __shared__ float scratch[N][M];
        
        MulImpl<Transpose>(*X, *v, *w, scratch);
    }

    template<typename ErrType, typename RefType>
    __host__ void CheckErrorThreshold(float errVal, float threshold, const ErrType& errTensor, const RefType& refTensor, const char* message, int& errorCount, const bool verbose)
    {
        if (errVal > threshold)
        {
            printf_red("%s: FAILED (error %f)\n", message, errVal);
            std::printf("Error:\n");
            errTensor.Print();
            std::printf("Reference:\n");
            refTensor.Print();
            std::printf("\n");
            
            errorCount++;
        }
        else if(verbose)
        {
            printf_green("%s: PASSED!\n", message);
        }
    }

    __host__ void TestSquare8x8TensorMul(const bool verbose, int& errorCount)
    {
        constexpr float kErrorThreshold = 1e-6;
        constexpr int N = 8;

        Cuda::Object<Tensor2D<N, N, false>> X;
        Cuda::Object<Tensor1D<N, false>> v;
        Cuda::Object<Tensor1D<N, false>> r;

        // NOTE: Row-major order constuctor
        X <<= Tensor2D<N, N, false>({ {0.652467807974029, 0.633070356251368, 0.68281308686666,
                                      0.566351831093323, 0.935202196659332, 0.976187756902101,
                                      0.238451694824191, 0.637562295790242}, {0.101098380420291,
                                      0.64552469382196, 0.159522225810158, 0.813787851275935,
                                      0.904785470451441, 0.640712457447006, 0.306539727511699,
                                      0.756197597784415}, {0.876688377753995, 0.019128400638492,
                                      0.542616681289372, 0.352370872609403, 0.899199876332835,
                                      0.968878409682844, 0.876215073712776,
                                      0.340281445182268}, {0.282418445376708, 0.296477088550717,
                                      0.695952684977098, 0.514103363052473, 0.781133951426663,
                                      0.403595994418056, 0.38515245011824,
                                      0.379794153435092}, {0.43271249281167, 0.123592882524315,
                                      0.692030134942357, 0.941189043995746, 0.0907422167282328,
                                      0.816402708208848, 0.956728218285361,
                                      0.639900147904955}, {0.957129994567048, 0.635376315590379,
                                      0.490904095442375, 0.37038068606824, 0.314276772143277,
                                      0.65659249717978, 0.301137853531729,
                                      0.847904621825146}, {0.0226571509809761, 0.163593478192408,
                                      0.694356353588249, 0.766335669320467, 0.210146094825781,
                                      0.69231648501517, 0.423810127960657,
                                      0.30261765393665}, {0.0943698123927186, 0.796860647653511,
                                      0.0290345835834827, 0.361089338620035, 0.294762947477966,
                                      0.191027881734111, 0.74805085393239, 0.549756115833064} });

        // Input vector
        v <<= Tensor1D<N, false>({ {0.745997562146465}, {0.490766766044425}, {0.492705416561543}, {0.52926641341919}, {0.349661831500469}, {0.515445231303594}, {0.341862603700548}, {0.310267848957651} });

        // Targets for multiply and multiply transpose
        const Tensor1D<N, false> targetMul({ {2.54311463071826}, {1.88756865074411}, {2.33618641198045}, {1.70185336760994}, {2.20071450422554}, {2.27809276352237}, {1.51400595076495}, {1.29472435169277} });
        const Tensor1D<N, false> targetMulT({ {1.79945551487407}, {1.62929517783041}, {1.96475333274066}, {2.16161456090267}, {2.35518392263892}, {2.65350477631207}, {1.83062043735004}, {2.15022972677939} });

        // Test and check errors
        KernelMulSquare<false> << <1, N* N >> > (X.GetDeviceData(), v.GetDeviceData(), r.GetDeviceData());
        IsOk(cudaDeviceSynchronize());
        r.Download();
        const float errorMul = CwiseMax(Abs(*r - targetMul));
        CheckErrorThreshold(errorMul, kErrorThreshold, *r, targetMul, "TestSquare8x8TensorMul: mul", errorCount, verbose);

        KernelMulSquare<true> << <1, N* N >> > (X.GetDeviceData(), v.GetDeviceData(), r.GetDeviceData());
        IsOk(cudaDeviceSynchronize());
        r.Download();
        const float errorMulT = CwiseMax(Abs(*r - targetMulT));
        CheckErrorThreshold(errorMulT, kErrorThreshold, *r, targetMulT, "TestSquare8x8TensorMul: mul transpose", errorCount, verbose);
    }


    __host__ void TestSquare4x4TensorMul(const bool verbose, int& errorCount)
    {
        constexpr float kErrorThreshold = 1e-6;
        constexpr int N = 4;
        
        Cuda::Object<Tensor2D<N, N, false>> X;
        Cuda::Object<Tensor1D<N, false>> v;
        Cuda::Object<Tensor1D<N, false>> r;

        // NOTE: Row-major order constuctor
        X <<= Tensor2D<N, N, false>({ {0.652467807974029, 0.633070356251368, 0.68281308686666, 0.566351831093323},
                                    {0.935202196659332, 0.976187756902101, 0.238451694824191, 0.637562295790242},
                                    {0.101098380420291, 0.64552469382196, 0.159522225810158, 0.813787851275935},
                                    {0.904785470451441, 0.640712457447006, 0.306539727511699, 0.756197597784415} });

        // Input vector
        v <<= Tensor1D<N, false>({ 0.876688377753995, 0.019128400638492, 0.542616681289372, 0.352370872609403 });

        // Targets for multiply and multiply transpose
        const Tensor1D<N, false> targetMul({ {1.15419222757901}, {1.19260005697745}, {0.474294186123722}, {1.23826628790775} });
        const Tensor1D<N, false> targetMulT( {{0.963577579819826}, {1.1497192089129}, {0.797750589019602}, {1.21674648559447}});

        // Test and check errors
        KernelMulSquare<false> << <1, N* N >> > (X.GetDeviceData(), v.GetDeviceData(), r.GetDeviceData());
        IsOk(cudaDeviceSynchronize());
        r.Download();
        const float errorMul = CwiseMax(Abs(*r - targetMul));
        CheckErrorThreshold(errorMul, kErrorThreshold, *r, targetMul, "TestSquare4x4TensorMul: mul", errorCount, verbose);

        KernelMulSquare<true> << <1, N*N >> > (X.GetDeviceData(), v.GetDeviceData(), r.GetDeviceData());
        IsOk(cudaDeviceSynchronize());
        r.Download();
        const float errorMulT = CwiseMax(Abs(*r - targetMulT));
        CheckErrorThreshold(errorMulT, kErrorThreshold, *r, targetMulT, "TestSquare4x4TensorMul: mul transpose", errorCount, verbose);     
    }    

    __host__ void TestNonSquareTensorMul(const bool verbose, int& errorCount)
    {
        constexpr float kErrorThreshold = 1e-6;
        constexpr int N = 7, M = 3;

        Cuda::Object<Tensor2D<N, M, false>> X;
        Cuda::Object<Tensor1D<N, false>> v, rw;
        Cuda::Object<Tensor1D<M, false>> w, rv;

        // NOTE: Row-major order constuctor
        X <<= Tensor2D<N, M, false>({ {0.817389490171071, 0.111419611131236, 0.789525994633852, 0.187803146706026, 0.24136096745765, 0.0657387595087811, 0.542246620509624},
                                    {0.231154506736027, 0.396006081548587, 0.700473781942245, 0.211825979054127, 0.748656881482948, 0.422850649339949, 0.247494780864008},
                                    {0.977171761740777, 0.825162939488514, 0.925275201178863, 0.578056151994379, 0.292869736797923, 0.208051064780593, 0.580474481294159} });

        // Input vector
        v <<= Tensor1D<N, false>({ {0.128820834638437}, {0.306427380034618}, {0.712012070075779}, {0.390581871380074}, {0.81996723678518}, {0.325351497995291}, {0.593260227507795} });
        w <<= Tensor1D<M, false>({ {0.518774167040164}, {0.16901301687775}, {0.472565143196439} });

        // Targets for multiply and multiply transpose
        const Tensor1D<M, false> targetMul({ {1.31593300106679}, {1.63088381395845}, {1.91552370181268} });
        const Tensor1D<N, false> targetMulT({ {0.924885985973772}, {0.514675041160793}, {0.965227685293762}, {0.406397957015768}, {0.390144622102385}, {0.203888515360294}, {0.5974453848352} });

        //X = Tensor2D<N, M>({ {1., 1., 1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1., 1., 1.} });
        //v = Tensor1D<N>({ {1.},{1.},{1.},{1.},{1.},{1.},{1.} });
        //w = Tensor1D<M>({ {1.},{1.},{1.} });
        //const Tensor1D<M> targetMul({ {7.},{7.},{7.} });
        //const Tensor1D<N> targetMulT({ {3.},{3.},{3.},{3.},{3.},{3.},{3.} });

        // Test and check errors
        KernelMulNonSquare<false> << <1, N* M >> > (X.GetDeviceData(), v.GetDeviceData(), rv.GetDeviceData());
        IsOk(cudaDeviceSynchronize());
        rv.Download();
        const float errorMul = CwiseMax(Abs(*rv - targetMul));
        CheckErrorThreshold(errorMul, kErrorThreshold, *rv, targetMul, "TestNonSquareTensorMul: mul", errorCount, verbose);

        KernelMulNonSquare<true> << <1, N*M >> > (X.GetDeviceData(), w.GetDeviceData(), rw.GetDeviceData());
        IsOk(cudaDeviceSynchronize());
        rw.Download();
        const float errorMulT = CwiseMax(Abs(*rw - targetMulT));
        CheckErrorThreshold(errorMulT, kErrorThreshold, *rw, targetMulT, "TestNonSquareTensorMul: mul transpose", errorCount, verbose);
    }

    __host__ void RunTensorTests(const bool verbose)
    {
        int errorCount = 0;
        
        TestSquare4x4TensorMul(verbose, errorCount);
        TestSquare8x8TensorMul(verbose, errorCount);
        TestNonSquareTensorMul(verbose, errorCount);

        AssertFmt(errorCount == 0, "Test failed with %i errors", errorCount);

        if (verbose) { printf_green("All tensor tests okay!\n"); }
    }

}