#pragma once

#include "../Includes.h"
#include <fstream>

namespace Flair
{
    namespace IO
    {
        template<template<typename, typename...> typename ContainerType, typename Type, typename... Pack>
        size_t SerialiseArray(const ContainerType<Type, Pack...>& data, const char* filePath)
        {
            if (data.size() == 0) { return 0; }

            std::ofstream file(filePath, std::ios::out | std::ios::binary);
            Assert(file.is_open());

            file.write(reinterpret_cast<const char*>(data.data()), sizeof(Type) * data.size());
            return sizeof(Type) * data.size();
        }

        template<template<typename, typename...> typename ContainerType, typename Type, typename... Pack>
        size_t DeserialiseArray(ContainerType<Type, Pack...>& data, const char* filePath)
        {
            std::ifstream file(filePath, std::ios::in | std::ios::binary);
            AssertFmt(file.is_open(), "Error: '%s' not found.", filePath);

            file.seekg(0, std::ios::end);
            const size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            AssertFmt(fileSize % sizeof(Type) == 0, "Error: number of bytes in file '%s' is not a multiple of %s size (%i)", typeid(Type).name(), sizeof(Type));

            data.resize(fileSize / sizeof(Type));
            file.read(reinterpret_cast<char*>(data.data()), fileSize);

            return fileSize;
        }
    }
}