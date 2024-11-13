#pragma once

#include "CoderUtils.h" 
#include "core/math/MathUtils.h"

namespace Flair
{

    template<typename InputType, typename OutputType = uint8_t>
    class ArithmeticCoder
    {
    public:
        using Type = InputType;
        using Model = std::vector<std::pair<InputType, uint32_t>>;

    private:
        struct PMF
        {
            static size_t SizeOf() { return sizeof(InputType) + sizeof(uint32_t); }

            InputType  c;
            uint32_t    pmf;
            uint32_t    cmf;
        };

        static constexpr uint32_t kModelPrecision = 16;
        static constexpr uint32_t kModelMass = (1 << kModelPrecision) - 1;
        static constexpr uint32_t kModelMinMass = 1 << 0;
        static constexpr uint32_t kIMax = 0xffffffff;
        static constexpr uint16_t kJMax = 0xffff;

        std::vector<PMF>    m_model;

    public:
        ArithmeticCoder() = default;
        ArithmeticCoder(const std::vector<InputType>& input)
        {
            Build(input);
        }

        float ComputeShannonEntropy() const
        {
            double entropy = 0;
            for (auto it = std::next(m_model.begin()); it != m_model.end(); ++it)
            {
                const double P = it->pmf / double(kModelMass);
                entropy += P * std::log(P) / kLog2;
            }

            return float(-entropy);
        }

        int ComputeBitsPerSymbol() const
        {
            return std::ceil(std::log(float(m_model.size() - 1)) / float(kLog2));
        }

        float ComputeCodingEfficiency(const std::vector<OutputType>& encoded, const int messageLength) const
        {
            const float nullEntropy = ComputeBitsPerSymbol() * messageLength;
            const float shannonEntropy = ComputeShannonEntropy() * messageLength;
            const float encodedEntropy = encoded.size() * sizeof(OutputType) * 8;

            return (encodedEntropy - nullEntropy) / (shannonEntropy - nullEntropy);
        }

        Model GetModel() const
        {
            Model output;
            output.reserve(m_model.size());
            for (auto& element : m_model)
            {
                output.emplace_back(element.c, element.pmf);
            }
            return output;
        }

        size_t GetModelSize() const { return m_model.size() * PMF::SizeOf(); }

        void SetModel(const Model& model)
        {
            Assert(model.size() >= 2);

            // Copy the PMF
            m_model.clear();
            m_model.reserve(model.size());
            for (auto& element : model)
            {
                m_model.push_back({ element.first, element.second, 0 });
            }

            // Build the CMF
            uint32_t cmf = 0;
            for (auto element = std::next(m_model.begin()); element != m_model.end(); ++element)
            {
                element->cmf = cmf + element->pmf;
                cmf = element->cmf;
            }
        }

        template<typename ContainerType>
        void Build(const ContainerType& container, const bool debug = false)
        {
            Assert(!container.empty());
            Assert(container.size() < (kModelMass - 1) / kModelMinMass);

            // Prime the PMF
            m_model.clear();
            m_model.push_back({ ' ', 0, 0 });  // DON'T USE THIS WHEN SORTING BY LARGEST IN THE MIDDLE

            // Load the frequencies into the model
            uint64_t sumMass = 0;
            for (const auto& element : container)
            {
                m_model.push_back({ element.first, uint32_t(element.second), 0 });
                sumMass += element.second;
            }

            // Normalise the masses
            int sumNormMass = 0;
            int sumNormMassGtMin = 0;
            for (auto element = std::next(m_model.begin()); element != m_model.end(); ++element)
            {
                const uint64_t adjustedPmf = uint64_t(element->pmf) * kModelMass / sumMass;
                element->pmf = std::max(kModelMinMass, uint32_t(adjustedPmf));

                sumNormMass += element->pmf;
                if (element->pmf > kModelMinMass) { sumNormMassGtMin += element->pmf; }
            }

            // Sort by descending frequency
            std::sort(std::next(m_model.begin()), m_model.end(), [](const PMF& a, const PMF& b) { return a.pmf > b.pmf; });

            if (debug)
            {
                std::printf("sumMass: %i\n", int(sumMass));
                std::printf("sumNormMass: %i\n", sumNormMass);
                std::printf("sumNormMassGtMin: %i\n", sumNormMassGtMin);
            }

            // If the sum of the normalised mass doesn't equal the target mass, distribute the error across each element according to its magnitude
            std::vector<int> changes;
            if (sumNormMass != kModelMass)
            {
                const int deltaMass = sumNormMass - kModelMass;
                int cmf = 0;
                for (auto& element : m_model)
                {
                    const int prevPmf = element.pmf;
                    if (deltaMass > 0 && element.pmf > kModelMinMass)
                    {
                        // If we're removing mass, don't remove it from elements of whose sizes equal the minimum
                        element.pmf -= (1 + 2 * deltaMass * (cmf + int(element.pmf))) / (2 * sumNormMassGtMin) -
                            (1 + 2 * deltaMass * cmf) / (2 * sumNormMassGtMin);
                    }
                    else if (deltaMass < 0)
                    {
                        element.pmf += (1 + 2 * deltaMass * cmf) / (2 * sumNormMass) -
                            (1 + 2 * deltaMass * (cmf + int(element.pmf))) / (2 * sumNormMass);
                    }
                    cmf += prevPmf;
                    changes.push_back(int(element.pmf) - int(prevPmf));
                }

                // Re-sort by descending frequency
                std::sort(std::next(m_model.begin()), m_model.end(), [](const PMF& a, const PMF& b) { return a.pmf > b.pmf; });
            }

            // Build the CMF and run checks
            uint32_t cmf = 0;
            for (auto element = std::next(m_model.begin()); element != m_model.end(); ++element)
            {
                // FIXME: Very large models can cause overflows and fail. This is very bad!
                if (element->pmf < kModelMinMass)
                {
                    //PrintModel();

                    std::printf("sumMass: %i\n", int(sumMass));
                    std::printf("kModelMass: %i\n", kModelMass);
                    std::printf("sumNormMass: %i\n", sumNormMass);
                    std::printf("sumNormMassGtMin: %i\n", sumNormMassGtMin);
                    std::printf("deltaMass: %i\n", sumNormMass - kModelMass);

                    //for (auto i : changes) std::printf("%i ", i); std::printf("\n");

                    AssertMsg(false, "Element is smaller than minimum mass.");
                }

                element->pmf <<= (15 - (kModelPrecision - 1));
                element->cmf = cmf + element->pmf;
                cmf = element->cmf;
            }

            /*std::deque<PMF> pmfDeque;
            for(int i = 0; i < m_model.size(); i++)
            {
                if(i % 2)
                {
                    pmfDeque.push_back(m_model[i]);
                }
                else
                {
                    pmfDeque.push_front(m_model[i]);
                }
            }

            m_model.clear();
            m_model.push_back({' ', 0.0f, 0.0f, 0u});
            for(auto& f : pmfDeque) { m_model.push_back(f); */

            Assert(m_model.size() >= 2);

            if (debug) { PrintModel(); }
        }

        void Build(const std::vector<InputType>& input, const bool debug = false)
        {
            AssertMsg(!input.empty(), "Input is empty");

            // Count the frequency of occurances of each character
            std::unordered_map<InputType, int> frequencyMap;
            for (const auto& element : input)
            {
                auto it = frequencyMap.find(element);
                if (it == frequencyMap.end())
                {
                    frequencyMap[element] = 1;
                }
                else
                {
                    it->second++;
                }
            }

            if (debug) std::printf("Building model on %zi elements...\n", input.size());

            Build(frequencyMap, debug);
        }

        void PrintModel() const
        {
            for (auto& f : m_model)
            {
                std::printf("0x%8x (%c) - P: %s (%f) C: %s (%f)\n", f.c, char(f.c), bstr(f.pmf).c_str(), f.pmf / float(kJMax), bstr(f.cmf).c_str(), f.cmf / float(kJMax));
            }
            std::printf("\n");
        }

        std::vector<OutputType> Encode(const std::vector<InputType>& input, const bool isDebug = false, const int monitor0 = 0, const int monitor1 = std::numeric_limits<int>::max())
        {  
            auto debug = [&](const int x)
            {
                return isDebug && x >= monitor0 && x <= monitor1;
            };

            // Model not defined
            AssertMsg(m_model.size() >= 2, "Model is not defined.");            

            std::vector<OutputType> encoded;
            int bitIdx = -1;
            uint32_t i0 = 0x0;
            uint32_t i1 = kIMax;
            uint32_t underflowDepth = 0;

            // Reserve 4 bytes at the beginning of the encoded stream to store the size of the message
            encoded.resize(sizeof(uint32_t) / sizeof(OutputType));
            Assert(!encoded.empty() && input.size() < std::numeric_limits<uint32_t>::max()); // Sanity check
            const uint32_t inputSize = input.size();
            memcpy(encoded.data(), &inputSize, sizeof(uint32_t));

            // If the model only has one element then all the characters of the string are the same
            if (m_model.size() == 2) 
            { 
                encoded.push_back(1ul << (sizeof(OutputType) * 8 - 1));
                return encoded;
            }

            for (int msgIdx = 0; msgIdx < input.size(); msgIdx++)
            {
                const InputType cur = input[msgIdx];

                if (i0 == i1)
                {
                    std::printf("Under: i1: %s (%.10f)\n       i0: %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax));
                    AssertMsg(false, "Range is zero");
                }

                uint32_t j0, j1;
                int pmfIdx;
                for (pmfIdx = 1; pmfIdx < m_model.size(); pmfIdx++)
                {
                    if (m_model[pmfIdx].c == cur)
                    {
                        j0 = m_model[pmfIdx - 1].cmf;
                        j1 = m_model[pmfIdx].cmf;
                        break;
                    }
                }
                AssertMsg(pmfIdx < m_model.size(), "Input contained a symbol that was not in the model.");

                const uint32_t di = i1 - i0;
                const uint32_t diHi = di >> 16ul;
                const uint32_t diLo = di & 0xffff;

                if (debug(msgIdx))
                {
                    std::printf("\nIteration %i ('%c')\n", msgIdx, char(cur));
                    std::printf("PMF i: [%i, %i]\n", pmfIdx - 1, pmfIdx);
                    std::printf("Model: j1: %s (%.10f)\n       j0: %s (%.10f)\n", bstr<uint16_t>(j1).c_str(), float(j1) / float(kJMax), bstr<uint16_t>(j0).c_str(), float(j0) / float(kJMax));
                    std::printf("Intvl: i1: %s (%.10f)\n       i0: %s (%.10f)\n       d:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(di).c_str(), float(di) / float(kIMax));
                }

                i1 = i0 + (diHi * j1) + ((diLo * j1) >> 16ul);
                i0 = i0 + (diHi * j0) + ((diLo * j0) >> 16ul);

                if (debug(msgIdx))
                {
                    std::printf("Shrnk: i1: %s (%.10f)\n       i0: %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax));
                }

                // If we're in an undeflow state...
                if (((i1 >> 30ul) & 3ul) == 2ul && ((i0 >> 30ul) & 3ul) == 1ul)
                {
                    uint32_t shift = 0ul;
                    for (shift = 0ul; shift <= 29ul; shift++)
                    {
                        const uint32_t bitmask = 1ul << (29ul - shift);
                        if ((i0 & bitmask) == 0ul || (i1 & bitmask) != 0ul) { break; }
                    }
                    if (shift == 29ul)
                    {
                        std::printf("Under: i1: %s (%.10f)\n       i0: %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax));
                        PrintModel();
                        AssertMsg(false, "Underflow failure");
                    }

                    shift++;
                    i0 = (i0 << shift) ^ (1ul << 31ul);
                    i1 = (i1 << shift) ^ (1ul << 31ul);

                    underflowDepth += shift;

                    if (debug(msgIdx))
                    {
                        std::printf("Under: i1: %s (%.10f)\n       i0: %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax));
                        std::printf("Depth: %i\n", underflowDepth);
                    }
                }
                else
                {
                    uint32_t leading;
                    std::string emitted;
                    for (leading = 0ul; leading < 31ul; leading++)
                    {
                        const uint32_t bitmask = 1ul << (31ul - leading);

                        if ((i0 & bitmask) != (i1 & bitmask)) { break; }

                        if (underflowDepth)
                        {
                            // If we're in an underflow state, output the padded bits.
                            // If the leading bit is 0, output 01+. Otherwise, output 10+. Here, + indicates the padding.
                            PackBit(encoded, bitIdx, (i0 >> 31) & 1ul, &emitted);
                            for (; underflowDepth > 0; underflowDepth--)
                            {
                                PackBit(encoded, bitIdx, (~i0 >> 31) & 1ul, &emitted);
                            }
                        }
                        else
                        {
                            PackBit(encoded, bitIdx, (i0 >> (31ul - leading)) & 1ul, &emitted);
                        }
                    }
                    if (leading == 31ul)
                    {
                        std::printf("Intvl: i1: %s (%.10f)\n       i0: %s (%.10f)\n       d:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(di).c_str(), float(di) / float(kIMax));
                        AssertMsg(leading < 31u, "Leading overflow.");
                    }

                    if (leading > 0ul)
                    {
                        if (isDebug) std::printf("Out: %s -> 0.%s (%i)\n", emitted.c_str(), bstr(encoded.empty() ? 0ul : encoded.back()).c_str(), leading);

                        // Renormalise by shifting and masking. Make sure to preserve the MSB of i1 in case it's 1.0000
                        i0 <<= leading;
                        i1 <<= leading;

                        if (debug(msgIdx)) std::printf("Renor: i1: %s (%.10f)\n       i0: %s (%.10f) (%i bits)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), leading);
                    }
                }
            }

            // Terminator
            PackBit(encoded, bitIdx, 1ul);

            return encoded;
        }

        std::vector<InputType> Decode(const std::vector<OutputType>& encoded, const bool isDebug = false, const int monitor0 = 0, const int monitor1 = std::numeric_limits<int>::max())
        {
            auto debug = [&](const int x)
            {
                return isDebug && x >= monitor0 && x <= monitor1;
            };

            // The first 4 bytes of the encoded stream contain the length of the uncompressed buffer
            uint32_t length = 0;
            if (encoded.size() >= sizeof(uint32_t) / sizeof(OutputType))
            {
                memcpy(&length, encoded.data(), sizeof(uint32_t));
            }

            AssertMsg(length > 0, "Encoded input stream is empty.");

            std::vector<InputType> decoded;
            decoded.reserve(length);

            uint32_t f = 0ul;
            uint32_t i0 = 0ul;
            uint32_t i1 = kIMax;
            int bitIdx = -1;
            uint32_t elementIdx = sizeof(uint32_t) / sizeof(OutputType);

            // Prime the 32 most significant bits with the contents of the encoded buffer        
            switch (sizeof(OutputType))
            {
            case 1:
                f = (uint32_t(encoded[elementIdx]) << 24) |
                    (uint32_t((encoded.size() - elementIdx > 1) ? encoded[elementIdx + 1] : 0) << 16) |
                    (uint32_t((encoded.size() - elementIdx > 2) ? encoded[elementIdx + 2] : 0) << 8) |
                    (uint32_t(encoded.size() - elementIdx > 3) ? encoded[elementIdx + 3] : 0);
                break;
            case 2:
                f = uint16_t(encoded[elementIdx] << 16) |
                    uint16_t((encoded.size() - elementIdx > 1) ? encoded[elementIdx + 1] : 0);
                break;
            default:
                f = uint32_t(encoded[elementIdx]);
            }

            elementIdx += 4 / sizeof(OutputType) - 1;

            if (isDebug) std::printf("\n\033[33mDecode %zi bytes: %s...\033[39m\n", encoded.size() * 4, packed_bstr(encoded).c_str());

            for (int msgIdx = 0; msgIdx < length; msgIdx++)
            {
                if (debug(msgIdx))
                {
                    std::printf("\nIteration %i:\n", msgIdx);
                    std::printf("Register: %s\n", bstr(f).c_str());
                    std::printf("Idx:   %i -> %zi\n", elementIdx, encoded.size() - 1);
                }

                const uint32_t di = i1 - i0;
                const uint32_t diHi = di >> 16;
                const uint32_t diLo = di & 0xffff;

                /*if (debug(msgIdx) && msgIdx == monitor1)
                {
                    std::printf("Search: %f\n", f / double(kIMax));
                    for (int k = 0; k < m_model.size() - 1; k++)
                    {
                        const uint32_t fMid0 = i0 + (diHi * m_model[k].cmf) + ((diLo * m_model[k].cmf) >> 16ul);
                        const uint32_t fMid1 = i0 + (diHi * m_model[k+1].cmf) + ((diLo * m_model[k+1].cmf) >> 16ul);
                        std::printf(" - [%i, %i) -> [%s, %s) [%f, %f)", k, k+1, bstr(fMid0).c_str(), bstr(fMid1).c_str(), fMid0 / double(kIMax), fMid1 / double(kIMax));
                        if (f >= fMid0 && f < fMid1) { std::printf(" <-----------"); }
                        std::printf("\n");
                    }
                }*/
                int j0 = 0, j1 = m_model.size() - 1;
                while (j1 - j0 > 0)
                {
                    const int jMid = j0 + (j1 - j0) / 2;
                    const uint32_t fMid = i0 + (diHi * m_model[jMid].cmf) + ((diLo * m_model[jMid].cmf) >> 16ul);
                    if (fMid <= f)
                    {
                        j0 = jMid + 1; // Go right
                    }
                    else
                    {
                        j1 = jMid; // Go left
                    }
                }
                if (j1 == 0) { j1 = 1; }

                if (debug(msgIdx))
                {
                    std::printf("PMF i: [%i, %i]\n", j1 - 1, j1);
                    std::printf("Model: j1: %s (%.10f)\n       j0: %s (%.10f)\n", bstr(m_model[j1].cmf).c_str(), float(m_model[j1].cmf) / float(kJMax), bstr(m_model[j1 - 1].cmf).c_str(), float(m_model[j1 - 1].cmf) / float(kJMax));
                    std::printf("Intvl: i1: %s (%.10f)\n       i0: %s (%.10f)\n       d:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(di).c_str(), float(di) / float(kIMax));
                }

                i1 = i0 + (diHi * m_model[j1].cmf) + ((diLo * m_model[j1].cmf) >> 16ul);
                i0 = i0 + (diHi * m_model[j1 - 1].cmf) + ((diLo * m_model[j1 - 1].cmf) >> 16ul);

                // Push the character to the decoded output
                decoded.push_back(m_model[j1].c);

                if (debug(msgIdx))
                {
                    std::printf("Pushed 0x%x ('%c')\n", m_model[j1].c, m_model[j1].c);
                    std::printf("Shrnk: i1: %s (%.10f)\n       i0: %s (%.10f)\n       f:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(f).c_str(), float(f) / float(kIMax));
                }

                // Underflow condition
                if (((i1 >> 30ul) & 3ul) == 2ul && ((i0 >> 30ul) & 3ul) == 1ul)
                {
                    uint32_t nudge;
                    for (nudge = 0ul; nudge <= 29ul; nudge++)
                    {
                        const uint32_t bitmask = 1ul << (29ul - nudge);
                        if ((i0 & bitmask) == 0ul || (i1 & bitmask) != 0ul) { break; }
                    }
                    if (nudge == 29ul)
                    {
                        std::printf("Intvl: i1: %s (%.10f)\n       i0: %s (%.10f)\n       f:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(f).c_str(), float(f) / float(kIMax));
                        PrintModel();
                        AssertMsg(false, "Underflow failure");
                    }

                    nudge++;

                    if (debug(msgIdx)) std::printf("Nudge: %i\n", nudge);

                    f = (f << nudge) ^ (1ul << 31ul);
                    i0 = (i0 << nudge) ^ (1ul << 31ul);
                    i1 = (i1 << nudge) ^ (1ul << 31ul);

                    for (int shift = nudge - 1; shift >= 0; shift--)
                    {
                        f |= UnpackBit(encoded, elementIdx, bitIdx) << shift;
                    }

                    if (debug(msgIdx)) std::printf("Under: i1: %s (%.10f)\n       i0: %s (%.10f)\n       f:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(f).c_str(), float(f) / float(kIMax));
                }
                else
                {
                    uint32_t nudge;
                    for (nudge = 0ul; nudge < 31ul; nudge++)
                    {
                        const uint32_t bitmask = 1ul << (31ul - nudge);

                        if ((i0 & bitmask) != (i1 & bitmask)) { break; }
                    }
                    AssertMsg(nudge != 31ul, "Precision failure.");

                    if (nudge > 0ul)
                    {
                        // Renormalise by nudgeing and masking. Make sure to preserve the MSB of i1 in case it's 1.0000
                        f <<= nudge;
                        i0 <<= nudge;
                        i1 <<= nudge;

                        // Shift in more data
                        for (int shift = nudge - 1; shift >= 0; shift--)
                        {
                            f |= UnpackBit(encoded, elementIdx, bitIdx) << shift;
                        }

                        if (debug(msgIdx))
                        {
                            std::printf("Shift: %i (d: %i, b: %i)\n", nudge, elementIdx, bitIdx);
                            std::printf("Renor: i1: %s (%.10f)\n       i0: %s (%.10f)\n       f:  %s (%.10f)\n", bstr(i1).c_str(), float(i1) / float(kIMax), bstr(i0).c_str(), float(i0) / float(kIMax), bstr(f).c_str(), float(f) / float(kIMax));
                        }

                        /*if(debug(msgIdx))
                        {
                            const int b0 = 31 - bitIdx;
                            const int b1 = 32 + (31 - bitIdx);

                            for(int d = -1, b = 0; d <= 0; d++)
                            {
                                if(elementIdx + d < 0 || elementIdx + d >= encoded.size()) continue;

                                for(int i = 31; i >= 0; i--, b++)
                                {
                                    if(b == b0) std::printf("\033[33m");
                                    else if(b == b1) std::printf("\033[39m");
                                    if(b && b % 8 == 0) std::printf(",");

                                    std::printf("%c", ((encoded[elementIdx + d] >> i) & 1ul) + '0');
                                }
                            }
                            std::printf("\n");
                        }*/
                    }
                }
            }

            AssertFmt(decoded.size() == length, "Error: length of decoded data does not match header. Was %zi, should be %i", decoded.size(), length);
            return decoded;
        }

    private:
        inline void PackBit(std::vector<OutputType>& encoded, int& bitIdx, const uint32_t& value, std::string* emittedStr = nullptr)
        {
            // If the target element is full, push it to the encoded buffer
            if (bitIdx < 0)
            {
                encoded.push_back(0);
                bitIdx = sizeof(OutputType) * 8 - 1;
            }

            encoded.back() |= value << bitIdx;
            if (emittedStr) { *emittedStr += value + '0'; }
            bitIdx--;
        }

        inline uint32_t UnpackBit(const std::vector<OutputType>& encoded, uint32_t& elementIdx, int& bitIdx)
        {
            bool incd = false;
            if (bitIdx < 0)
            {
                if (elementIdx >= encoded.size() - 1) { return 0; }
                elementIdx++;
                incd = true;
                bitIdx = sizeof(OutputType) * 8 - 1;
            }

            return (encoded[elementIdx] >> bitIdx--) & 1ul;
        }
    };

    // FIXME: Why won't this compile?!
    /*template<typename InputType, typename OutputType>
    inline size_t SizeOf(const typename ArithmeticCoder<InputType, OutputType>::Model& model)
    {
        return model.size() * sizeof(typename ArithmeticCoder<InputType, OutputType>::Model::value_type);
    }*/

    inline size_t SizeOf(const ArithmeticCoder<uint16_t>::Model& model)
    {
        return model.size() * sizeof(ArithmeticCoder<uint16_t>::Model::value_type);
    }

}