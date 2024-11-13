
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

#include "CudaUtils.cuh"
#include "core/io/ImageIO.h"
#include "core/io/FilesystemUtils.h"
#include "core/math/MathUtils.h"
#include "core/utils/HighResTimer.h"
#include "core/utils/ConsoleUtils.h"
#include "core/analysis/Metrics.h"

#include "LiftingCodec.h"

#include <unordered_map>

namespace Flair
{
    __global__ void Test()
    {
        printf("%i\n", kKernelIdx);
    }

    void Run(int argc, char* argv[])
    {
        if (argc < 3)
        {
            std::printf("Usage: exr2flair [input (.exr)] [output (.exr)]\n");
            std::printf("  Params:\n"
                "  -euler=[x,y,z]      Comma-separated Euler angles in degrees of the HDRI reprojection. E.g. -euler=10,37.8,10.0\n"
                "  -diagnostics        Test the codec and outputs diagnostics to the same directory as the output file.\n"
                "  -verbose            Outputs additional debug information from the codec\n"
            );
            return;
        }

        std::printf("exr2flair:\n");

        // Parse the command line parameters
        std::string inputPath(argv[1]), outputPath(argv[2]);
        std::unordered_map<std::string, std::string> params;
        for (int i = 3; i < argc; ++i)
        {
            std::string arg(argv[i]);
            const size_t j = arg.find_last_of('=');
            params[arg.substr(1, j)] = (j != std::string::npos) ? arg.substr(j + 1, arg.length()) : "";
        }

        // Verify that the input and output paths are valid and the input file exists
        if (!FileExists(inputPath)) { std::printf("Input file '%s' not found.\n", inputPath.c_str()); return; }
        if (GetFileExtension(inputPath) != ".exr") { std::printf("Input file '%s' not found.\n", inputPath.c_str()); return; }
        if (!DirectoryExists(GetParentDirectory(outputPath))) { std::printf("Output directory for '%s' does not exist.\n", outputPath.c_str()); return; }
        const std::string outputExt = GetFileExtension(outputPath);
        if (outputExt != ".flair") { std::printf("Output file must either be .flair"); return; }

        const bool verbose = (params.find("verbose") != params.end());
        const bool diagnostics = (params.find("diagnostics") != params.end());     

        std::printf("Loading '%s'...\n", inputPath.c_str());
        Image3f inputImage;
        LoadEXR(inputPath, inputImage);

        Flair::LiftingCodec codec;

        // Encode the image
        HighResTimer wallTime;

        printf_green("Forward transform...\n");
        Flair::Image3f waveletImage = codec.Encode(inputImage);

        printf_yellow("Inverse transform...\n");
        Flair::Image3f decodedImage = codec.Decode(waveletImage);

        SaveEXR(ReplaceExtension(outputPath, ".wavelet.exr"), waveletImage);        
        SaveEXR(ReplaceExtension(outputPath, ".compressed.exr"), decodedImage);

        std::printf("Completed in %.2fs!\n", wallTime.Get());
    }
}

int main(int argc, char* argv[])
{
    try
    {
        IsOk(cudaSetDevice(0));
        
        Flair::Run(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        printf("Runtime error: %s\n", err.what());
    }

    return 0;
}

