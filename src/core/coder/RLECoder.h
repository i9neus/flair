#pragma once

#include "CoderUtils.h" 

namespace HDRI
{
    /*
    * Simple run-length coder for HDRI codec.
    * 
    * Encoder ingests a vector of integers and returns a reduced vector of RLE encoded integers plus a block table describing the layout of the data.
    * Encoder pass looks for contiguous blocks of a designated token (corresponding to quantised zero) larger than kMinRLEBlockSize. If found,
    * it creates an entry in the table signalling the length of the constant block. Other data is stored in a similar way, with table entries
    * indicating the run length of non-constant blocks.
    * 
    * Block format (assuming 16-bit table type)
    *   - Bit 31:    0 = non-const data, 1 = const data
    *   - Bits 0-30: Number of values in the block
    * 
    * */

    template<typename InputType>
    class RLECoder
    {
    public:        
        using TableType = uint16_t;
        using BlockTable = std::vector<TableType>;

    private:
        std::vector<uint16_t>               m_blocks;
        static constexpr int                kMinEncodingSize = 4;
        static constexpr InputType          kBlockTypeMask = (1 << (sizeof(TableType) * 8) - 1);
        static constexpr InputType          kBlockSizeMask = ~kBlockTypeMask;
        static constexpr InputType          kConstBlockFlag = kBlockTypeMask;
        static constexpr int                kMaxBlockLength = kBlockSizeMask;

    public:

        RLECoder() = delete;

        inline static bool IsConstBlock(const InputType value) { return value & kBlockTypeMask; }

        static float Encode(const std::vector<InputType>& input, const InputType rleToken, const int minRLEBlockSize, std::vector<InputType>& output, BlockTable& table)
        {
            table.clear();
            output.clear();
            
            if (input.size() < kMinEncodingSize)
            {
                output = input;
                return 1.;
            }

            for (int i = 0; i < input.size();)
            {               
                // If we've found an RLE token, check for possible spans we can compress...
                if (input[i] == rleToken)
                {                    
                    // Look ahead to determine how long this run is
                    int span = 1;
                    for (int j = i + 1; j < input.size() && input[j] == rleToken; ++j) { ++span; }

                    // Spans of 1 are just regular tokens, so only treat spans > 1 as runs.
                    if (span > 1)
                    {
                        // If the run is less than the minimum RLE block size, simply bulk-append the data to the output to save time
                        if (span < minRLEBlockSize)
                        {
                            output.resize(output.size() + span);
                            memcpy(&output[output.size() - span], &input[i], sizeof(InputType) * span);

                            if (table.empty()) { table.push_back(0); }
                            TableType& entry = table.back();
                            Assert(!IsConstBlock(entry)); // Sanity check

                            // If the block will overflow, fill it and start a new one with the residue
                            int blockSize = entry & kBlockSizeMask;
                            if (blockSize + span > kMaxBlockLength)
                            {
                                entry = kMaxBlockLength;
                                table.push_back(span - (kMaxBlockLength - blockSize));
                            }
                            else
                            {
                                entry += span;
                            }                         

                            i += span;
                            continue;
                        }
                        // Otherwise, the span of the constant block is larger than the minimum RLE size so we can encode it as a constant block
                        else
                        {
                            i += span;
                            
                            // Keep adding const blocks until we've exhausted the number of remaining tokens covered by the span
                            do
                            {
                                table.push_back(kConstBlockFlag | InputType(std::min(kMaxBlockLength, span)));
                                span -= kMaxBlockLength;
                            } 
                            while (span > 0);

                            continue;
                        }
                    }
                }               

                // Append the token to the output and update the last table entry
                if (table.empty()) { table.push_back(0); }
                TableType& entry = table.back();
                if (IsConstBlock(entry) || entry == kMaxBlockLength)
                {
                    table.push_back(1);
                }
                else
                {
                    ++entry;
                }
                output.push_back(input[i]);
                ++i;
            }
            
            AssertMsg(GetEncodedLength(table) == input.size(), "RLE encoder error: spanned data size and input size mismatch");

            // Return the compression factor of the coder
            return float(sizeof(InputType) * output.size() + sizeof(TableType) * table.size()) / float(sizeof(InputType) * input.size());
        }

        // Calculates the length of the encoding from a block table
        static size_t GetEncodedLength(BlockTable& table)
        {
            size_t sumSpan = 0;
            for (auto entry : table)
            {
                sumSpan += entry & kBlockSizeMask;
            }
            return sumSpan;
        }

        static void DynamicEncode(const std::vector<InputType>& input, const InputType rleToken, std::vector<InputType>& output, BlockTable& table)
        {
            int blockSizeLow = 2;
            int blockSizeHigh = int(std::sqrt(input.size())) / 4;
            int blockSizeMid = (blockSizeLow + blockSizeHigh) / 2;

            float ratioLow = Encode(input, rleToken, blockSizeLow, output, table);
            float ratioHigh = Encode(input, rleToken, blockSizeHigh, output, table);
            float ratioMid;

            while(std::abs(ratioLow - ratioHigh) > 0.02f && blockSizeHigh - blockSizeLow <= 4)
            {
                blockSizeMid = (blockSizeLow + blockSizeHigh) / 2;
                ratioMid = Encode(input, rleToken, blockSizeMid, output, table);

                if (ratioLow < ratioHigh)
                {
                    blockSizeHigh = blockSizeMid;
                    ratioHigh = ratioMid;
                }
                else
                {
                    blockSizeLow = blockSizeMid;
                    ratioLow = ratioMid;
                }
            }

            printf("RLE block size: %i\n", blockSizeMid);
        }

        static std::vector<InputType> Decode(const std::vector<InputType>& input, const InputType rleToken, const BlockTable& table)
        {
            // An empty table dictates that no compression has occurred
            if (table.empty()) { return input; }
            
            std::vector<InputType> output;
            int i = 0;
            for (const auto entry : table)
            {
                AssertMsg(entry != 0, "RLE decoder error: table entry should never be zero");
                
                const int blockSize = entry & kBlockSizeMask;

                // Resize the output buffer and pad with default token
                output.resize(output.size() + blockSize, rleToken);

                // If this entry indicates non-constant data, copy it from the input buffer
                if (!IsConstBlock(entry))
                {
                    AssertFmt(i + blockSize <= input.size(), "RLE decoder error: input buffer too small for block table.");
                    memcpy(&output[output.size() - blockSize], &input[i], blockSize * sizeof(InputType));
                    i += blockSize;
                }    
            }

            return output;
        }
    };
}