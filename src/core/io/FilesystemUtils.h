#pragma once

#include <filesystem>
#include <vector>

namespace Flair
{
    inline std::string ReplaceFilename(const std::string& absolutePath, const std::string& newFilename)
    {
        std::filesystem::path fsPath(absolutePath);
        return (!fsPath.has_filename() || !fsPath.has_stem()) ?
            std::string("") :
            fsPath.replace_filename(std::filesystem::path(newFilename)).string();
    }

    inline std::string ReplaceExtension(const std::string& absolutePath, const std::string& newExtension)
    {
        std::filesystem::path fsPath(absolutePath.c_str());
        return (!fsPath.has_extension() || !fsPath.has_stem()) ?
            std::string("") :
            fsPath.replace_extension(std::filesystem::path(newExtension.c_str())).string();
    }

    inline std::string GetParentDirectory(const std::string& absolutePath)
    {
        std::filesystem::path fsPath(absolutePath);
        return fsPath.has_parent_path() ? fsPath.parent_path().string() : "";
    }

    inline bool FileExists(const std::string& absolutePath)
    {
        if (absolutePath.empty()) { return false; }

        // Check to see whether the root path exists
        std::filesystem::path fsPath(absolutePath);
        return std::filesystem::exists(fsPath) && std::filesystem::is_regular_file(fsPath);
    }

    inline bool DirectoryExists(const std::string& absolutePath)
    {
        if (absolutePath.empty()) { return false; }

        // Check to see whether the root path exists
        std::filesystem::path fsPath(absolutePath);
        return std::filesystem::exists(fsPath) && std::filesystem::is_directory(fsPath);
    }

    inline std::string GetFileExtension(const std::string& absolutePath)
    {
        return std::filesystem::path(absolutePath).extension().string();
    }

    inline std::string GetFilename(const std::string& absolutePath)
    {
        return std::filesystem::path(absolutePath).filename().string();
    }

    inline std::vector<std::string> EnumerateDirectory(const std::string& dirPath)
    {
        std::vector<std::string> contents;
        namespace fs = std::filesystem;

        if (fs::is_directory(dirPath))
        {
            fs::directory_iterator end_iter;
            for (fs::directory_iterator dir_iter(dirPath); dir_iter != end_iter; ++dir_iter)
            {
                if (fs::is_regular_file(dir_iter->status()))
                {
                    contents.push_back(dir_iter->path().string());
                }
            }
        }

        return contents;
    }

    inline std::string Lowercase(const std::string& input)
    {
        std::string output = input;
        for (auto& c : output) { c = std::tolower(c); }
        return output;
    }
}
