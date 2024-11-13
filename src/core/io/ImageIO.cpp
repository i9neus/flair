#include "ImageIO.h"

#define STB_IMAGE_IMPLEMENTATION
#include "thirdparty/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb/stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include "thirdparty/tinyexr/tinyexr.h"

namespace Flair
{
    template<typename Type, int Channels>
    void LoadEXR(const std::string& path, Image<Type, Channels>& image)
    {
        static_assert(std::is_floating_point<Type>::value, "Image must be floating point type.");
        static_assert(Channels <= 3, "Image channels must be <= 3.");
        
        float* exrDataIn;
        int exrWidth, exrHeight;
        const char* exrErr = nullptr;
        if (::LoadEXR(&exrDataIn, &exrWidth, &exrHeight, path.c_str(), &exrErr) != TINYEXR_SUCCESS)
        {
            if (exrErr)
            {
                const std::string error = tfm::format("%s\n", exrErr);
                ::FreeEXRErrorMessage(exrErr);
                throw std::runtime_error(error);
            }
        }

        AssertMsg(exrWidth > 0 && exrHeight > 0 && exrDataIn, "tinyexr returned invalid values");

        image.Resize(exrWidth, exrHeight);
        image.Populate([&](const int x, const int y, float* pixel)
            {
                for (int c = 0; c < Channels; ++c)
                {
                    pixel[c] = exrDataIn[(y * exrWidth + x) * 4 + c];
                }
            });
        delete[] exrDataIn;
    }

    template void LoadEXR(const std::string& path, Image<float, 3>& image);
    
    /*template<typename Type, int Channels>
    void SaveEXR(Image<Type, Channels>& input, const std::string& filename)
    {
        EXRHeader header;
        InitEXRHeader(&header);

        EXRImage image;
        InitEXRImage(&image);

        image.num_channels = 3;

        std::vector<float> images[3];
        images[0].resize(width * height);
        images[1].resize(width * height);
        images[2].resize(width * height);

        // Split RGBRGBRGB... into R, G and B layer
        for (int i = 0; i < width * height; i++)
        {
            images[0][i] = rgb[3 * i + 0];
            images[1][i] = rgb[3 * i + 1];
            images[2][i] = rgb[3 * i + 2];
        }

        float* image_ptr[3];
        image_ptr[0] = &(images[2].at(0)); // B
        image_ptr[1] = &(images[1].at(0)); // G
        image_ptr[2] = &(images[0].at(0)); // R

        image.images = (unsigned char**)image_ptr;
        image.width = width;
        image.height = height;

        header.num_channels = 3;
        header.channels = (EXRChannelInfo*)std::malloc(sizeof(EXRChannelInfo) * header.num_channels);
        // Must be (A)BGR order, since most of EXR viewers expect this channel order.
        strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
        strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
        strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

        header.pixel_types = (int*)std::malloc(sizeof(int) * header.num_channels);
        header.requested_pixel_types = (int*)std::malloc(sizeof(int) * header.num_channels);
        for (int i = 0; i < header.num_channels; i++)
        {
            header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
            header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
        }
        header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

        const char* exrErr = NULL; // or nullptr in C++11 or later.
        int ret = SaveEXRImageToFile(&image, &header, outfilename.c_str(), &exrErr);
        if (ret != TINYEXR_SUCCESS)
        {
            const std::string error = tfm::format("Save EXR err: %s\n", exrErr);
            FreeEXRErrorMessage(exrErr);
            throw std::runtime_error(error);
        }
        std::printf("Saved exr file. [ %s ] \n", outfilename.c_str());

        std::free(header.channels);
        std::free(header.pixel_types);
        std::free(header.requested_pixel_types);
    }*/
}