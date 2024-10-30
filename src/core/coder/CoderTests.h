#pragma once

#include "ArithmeticCoder.h"
#include "RLECoder.h"

#include <random>

namespace HDRI
{
    template<typename T>
    int CompareEncodeDecode(const std::vector<T>& strA, const std::vector<T>& strB, bool translate = false)
    {
        constexpr int blockSize = 8;
        constexpr int numBlocks = 4;
        const std::vector<T>* strs[2] = { &strA, &strB };
        int idxs[2] = { 0, 0 };
        int errorIdx = -1;

        const std::string alphabet("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        std::map<T, char> translator;

        for (; idxs[0] < strA.size() || idxs[1] < strB.size();)
        {
            for (int pass = 0; pass < 2; pass++)
            {
                for (int c = 0; c < blockSize * numBlocks; c++)
                {
                    if (idxs[pass] >= strs[pass]->size()) { break; }

                    T thisValue = (*strs[pass])[idxs[pass] + c];

                    const int otherPass = (pass + 1) % 2;
                    if (idxs[pass] < strs[otherPass]->size() &&
                        thisValue != (*strs[otherPass])[idxs[pass] + c])
                    {
                        std::printf("\033[31m");

                        if (errorIdx == -1) { errorIdx = idxs[pass] + c; }
                    }

                    if ((idxs[pass] + c) % blockSize == 0 && (idxs[pass] + c) % (blockSize * numBlocks) != 0) { std::printf(" "); }

                    if (translate)
                    {
                        auto it = translator.find(thisValue);
                        if (it == translator.end())
                        {
                            translator[thisValue] = alphabet[translator.size() % alphabet.size()];
                            thisValue = translator[thisValue];
                        }
                        else
                        {
                            thisValue = it->second;
                        }
                    }

                    std::printf("%c", thisValue);

                    std::printf("\033[39m");
                }
                if (pass == 0) { std::printf("        "); }
            }

            idxs[0] += blockSize * numBlocks;
            idxs[1] += blockSize * numBlocks;
            std::printf("\n");
        }

        return errorIdx;
    }

    template<typename T, typename = typename std::enable_if_t<std::is_integral<T>::value>>
    static int VerifyDecodedData(const std::vector<T>& bufferA, const std::vector<T>& bufferB)
    {
        if (bufferA.size() != bufferB.size())
        {
            std::printf("Error decoding block: buffers are different sizes (%i -> %i)\n", bufferA.size(), bufferB.size());
            return std::min(bufferA.size(), bufferB.size());
        }

        int mismatchedByte = 0;
        for (int imismatchedByte = 0; mismatchedByte < bufferA.size(); mismatchedByte++)
        {
            if (bufferA[mismatchedByte] != bufferB[mismatchedByte]) { break; }
        }

        if (mismatchedByte >= bufferA.size()) { return -1; }

        std::printf("Error decoding block with size %i bytes. Byte: %i\n", bufferA.size(), mismatchedByte);

        std::printf("**** TABLE ****\n");
        std::printf("%i -> %i\n", bufferA.size(), bufferB.size());
        //CompareEncodeDecode(bufferA, bufferB, false);

        return mismatchedByte;
    }

    template<typename DataType>
    static std::vector<DataType> GenerateRLECoderDataset(const int inputSize, const float sparsity)
    {
        AssertMsg(sparsity >= 0 && sparsity <= 1, "Sparsity must be in the range [0, 1]");

        std::mt19937 mt(std::hash<int>{}(1));
        std::uniform_int_distribution<int> rng;

        std::vector<DataType> data(inputSize, 0);

        if (sparsity == 1.) { return data; }
        
        // Initialise the input array with random data
        for (auto& entry : data)
        {
            entry = DataType(rng(mt));
        }

        // Create voids with in the data until we've hit the target sparsity
        const int targetZeroes = std::min(inputSize, int(inputSize * sparsity));
        const int maxRange = std::max(1, inputSize / 10);
        constexpr int kNumMaxTries = 100000;
        int numTries = 0;
        for (int zeroCount = 0; zeroCount < targetZeroes && numTries < kNumMaxTries; ++numTries)
        {
            int range = std::max(1, int(maxRange * std::pow(float(rng(mt)) / float(std::numeric_limits<int>::max()), 5.f)));
            int startIdx = rng(mt) % inputSize;
            for (int i = 0; i < range && zeroCount < targetZeroes; ++i)
            {
                int j = (startIdx + i) % inputSize;
                if (data[j] != 0)
                {
                    data[j] = 0;
                    ++zeroCount;
                }
            }
        }
        Assert(numTries != kNumMaxTries);

        return data;
    }

    template<typename DataType>
    static void VerifyRLECoder()
    {        
        std::mt19937 mt(std::hash<int>{}(1));
        std::uniform_int_distribution<int> rng;

        constexpr int kNumRuns = 10;
        constexpr int kMinInputSize = 100;
        constexpr int kMaxInputSize = 100000;

        for (int testIdx = 0; testIdx < kNumRuns; ++testIdx)
        {
            const int inputSize = kMinInputSize + (rng(mt) % (kMaxInputSize - kMinInputSize));
            const float inputSparsity = float(rng(mt)) / float(std::numeric_limits<int>::max());

            std::printf("Test %i: %i entries, %f sparsity...\n", testIdx, inputSize, inputSparsity);
            
            std::vector<DataType> inputData = GenerateRLECoderDataset<DataType>(inputSize, inputSparsity);
            std::vector<DataType> encodedData;
            std::vector<DataType> outputData;
            RLECoder<DataType>::BlockTable table;

            //inputData = { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0 };

            RLECoder<DataType>::Encode(inputData, 0, 32, encodedData, table);
            RLECoder<DataType>::Decode(encodedData, 0, outputData, table);

            /*for (int i = 0; i < inputData.size(); ++i)
            {
                printf("%i: %i -> %i\n", i, inputData[i], outputData[i]);
            }*/

            VerifyDecodedData(inputData, outputData);

            printf("PASS!\n\n");
        }
    } 

    static void DetermineOptimumRLEBlockSize()
    {
        std::mt19937 mt(std::hash<int>{}(1));
        std::uniform_int_distribution<int> rng;

        using DataType = uint16_t;
        constexpr int kNumRuns = 10;
        constexpr int kMinInputSize = 100;
        constexpr int kMaxInputSize = 100000;
        constexpr float kMinSparsity = 0.8;
        constexpr float kMaxSparsity = 1.0;
        
        std::vector<float> compressionRatio(std::log2(4096) + 1, 0.f);

        for (int testIdx = 0; testIdx < kNumRuns; ++testIdx)
        {           
            const int inputSize = 1000;// kMinInputSize + (rng(mt) % (kMaxInputSize - kMinInputSize));
            const float inputSparsity = mix(kMinSparsity, kMaxSparsity, float(testIdx) / float(kNumRuns - 1));

            std::vector<DataType> inputData = GenerateRLECoderDataset<DataType>(inputSize, inputSparsity);
            std::vector<DataType> encodedData;
            RLECoder<DataType>::BlockTable table;

            for (int i = 1; i < inputData.size(); i += rng(mt) % 3) inputData[i] = 0;

            for (auto i : inputData) printf("%i ", i);
            printf("\n");
           
            std::printf("Sparsity: %f\n", inputSparsity);
            for (int blockIdx = 0, minBlockSize = 2; minBlockSize < 4096; minBlockSize *= 2, ++blockIdx)
            {
                table.clear();
                encodedData.clear();

                RLECoder<DataType>::Encode(inputData, 0, minBlockSize, encodedData, table);

                const int sizeIn = sizeof(DataType) * inputData.size();
                const int sizeOut = sizeof(DataType) * encodedData.size() + sizeof(RLECoder<DataType>::BlockTable::value_type) * table.size();

                float eff = float(sizeOut) / float(sizeIn);
                compressionRatio[blockIdx] += eff;            
                std::printf("  - %i: %.2f%% (%i -> %i)\n", minBlockSize, 100. * eff, sizeIn, sizeOut);
            }
            std::printf("\n");           
        }

        std::printf("RLE coding compressionRatio:\n");
        for (int blockIdx = 0, minBlockSize = 2; minBlockSize < 1024; minBlockSize *= 2, ++blockIdx)
        {
            std::printf("  - %i: %.2f%%\n", minBlockSize, 100. * compressionRatio[blockIdx] / kNumRuns);
        }
    }
}