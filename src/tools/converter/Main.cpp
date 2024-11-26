#include "core/io/ImageIO.h"
#include "core/io/FilesystemUtils.h"
#include "core/math/MathUtils.h"
#include "core/utils/HighResTimer.h"
#include "core/utils/ConsoleUtils.h"
#include "core/Codec.h"
#include "core/analysis/Metrics.h"

#include <unordered_map>

namespace Flair
{
    void Run(int argc, char* argv[])
    {
        if (argc < 3)
        {
            std::printf("Usage: converter [input (.exr)] [output (.exr)]\n");
            std::printf("  Params:\n"
                "  -euler=[x,y,z]      Comma-separated Euler angles in degrees of the HDRI reprojection. E.g. -euler=10,37.8,10.0\n"
                "  -diagnostics        Test the codec and outputs diagnostics to the same directory as the output file.\n"
                "  -verbose            Outputs additional debug information from the codec\n"
            );
            return;
        }

        std::printf("converter:\n");

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

        // Load the image. For now it's just .exrs.
        std::printf("Loading '%s'...\n", inputPath.c_str());
        Image3f inputImage;
        LoadEXR(inputPath, inputImage);

        std::printf("Okay!\n");
  
        uint32_t codecFlags = 0;
        if (verbose) { codecFlags |= Flair::kVerbose; }
        if (diagnostics) { codecFlags |= Flair::kOutputWaveletData; }

        Flair::Codec codec(codecFlags);

        // Encode the image
        HighResTimer wallTime;
        Flair::Image3f outputImage;
        Flair::CompressedImageData compressedData;
        codec.Encode(inputImage, compressedData);

        // Serialise the compressed image
        Flair::OutputFileStream outStream(outputPath);
        compressedData.Serialise(outStream);
        outStream.Close();
        printf_green("Compressed Flair image in %.2fs!\n", wallTime.Get());

        if (diagnostics)
        {
            std::printf("Running diagnostics...\n");

            // Load in the file we've just written out
            wallTime.Reset();
            Flair::InputStream inStream(outputPath);
            compressedData = Flair::CompressedImageData(inStream);

            // Decode the compressed file
            codec.Decode(compressedData, outputImage);
            printf_yellow("Decompressed Flair image in %.2fs!\n", wallTime.Get());

            // Generate and print some stats
            const auto stats = GenerateCodecStats(outputImage, inputImage, compressedData);
            Flair::PrintStats(stats);

            std::printf("Exporting additional data...\n");
            SaveEXR(ReplaceExtension(outputPath, ".wavelet.exr"), codec.GetWaveletData());
            SaveEXR(ReplaceExtension(outputPath, ".compressed.wavelet.exr"), outputImage);

            Flair::Image3f waveletCoeffs;
            codec.EncodeWaveletCoeffs(inputImage, waveletCoeffs);
            //codec.DecodeWaveletCoeffs(waveletCoeffs, outputImage);
            SaveEXR(ReplaceExtension(outputPath, ".compressed.exr"), outputImage);    
            SaveEXR(ReplaceExtension(outputPath, ".wavelet.exr"), codec.GetWaveletData());

            std::printf("Diagnostic checks complete!\n");
        }
    }

}

int main(int argc, char* argv[])
{
    try
    {
        Flair::Run(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        printf("Runtime error: %s\n", err.what());
    }

    return 0;
}

