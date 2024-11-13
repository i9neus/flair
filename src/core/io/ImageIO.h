#include "../Includes.h"
#include "../image/Image.h"

namespace Flair
{
    template<typename Type, int Channels>
    void LoadEXR(const std::string& path, Image<Type, Channels>& image);

    template<typename Type, int Channels>
    void SaveEXR(const std::string& path, const Image<Type, Channels>& image);
   
}