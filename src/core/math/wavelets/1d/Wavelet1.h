#pragma once

#include "core/Includes.h"

template<typename Wavelet, typename Real, int K0, int K1>
class Wavelet1
{
private:
	static constexpr int B0 = -K1 / 2;
	static constexpr int B1 = -(K0 - 1) / 2;

public:
	static void RunTests()
	{
		static bool isTested = false;
		if (isTested) { return; }
		isTested = true;

		std::vector<std::tuple<const std::string, const std::array<Real, 1 + K1 - K0>*, float>> data =
		{
			std::make_tuple(std::string("Forward mother"), &(typename Wavelet::ForwardFather()), Real(1)),
			std::make_tuple(std::string("Forward father"), &(typename Wavelet::ForwardMother()), Real(0)),
			std::make_tuple(std::string("Inverse mother"), &(typename Wavelet::InverseFather()), Real(1)),
			std::make_tuple(std::string("Inverse father"), &(typename Wavelet::InverseMother()), Real(0))
		};

		std::printf("Running wavelet tests...\n");
		std::printf("K0: %i   K1: %i\n", K0, K1);
		std::printf("B0: %i   B1: %i\n", B0, B1);

		constexpr Real epsilon = 1e-5;
		for (auto& arr : data)
		{
			Real sumWeight = 0;
			Real sumAbsWeight = 0;
			for (auto& i : *std::get<1>(arr))
			{
				sumWeight += i;
				sumAbsWeight += std::abs(i);
			}
			std::printf("%s:  %f  %f\n", std::get<0>(arr).c_str(), sumWeight, sumAbsWeight);
			AssertFmt(std::abs(sumWeight - std::get<2>(arr)) < epsilon, "%s weights sum to %f, should be %f.", std::get<0>(arr).c_str(), sumWeight, std::get<2>(arr));
		}
		std::printf("Passed!\n");
	}

	static int GetMinIOSize()
	{
		// The smallest dataset that can be transformed by this wavelet
		return 2 * std::max(1 - B0, 1 - B1);
	}

	static void Forward(const std::vector<Real>& input, std::vector<Real>& output, const int size)
	{
		//RunTests();

		AssertMsg((size & (size - 1)) == 0, "Not a power of 2.");

		const auto& father = Wavelet::ForwardFather();
		const auto& mother = Wavelet::ForwardMother();

		for (int i = 0, j = 0; i < size; i += 2, j++)
		{
			const int i0 = i + K0;
			const int i1 = i + K1;

			Real c0 = 0;
			Real c1 = 0;
			for (int p = i0, k = 0; p <= i1; p++, k++)
			{
				// Assume boundaries are reflected
				int pMod = p;
				if (pMod < 0) { pMod = -pMod; }
				else if (pMod >= size - 1) { pMod = 2 * (size - 1) - pMod; }

				c0 += input[pMod] * father[k];
				c1 += input[pMod] * mother[k];
			}
			output[j] = c0;
			output[j + size / 2] = c1;
		}
	}

	static void Inverse(const std::vector<Real>& input, std::vector<Real>& output, const int size)
	{
		AssertMsg((input.size() & (input.size() - 1)) == 0, "Not a power of 2.");

		const auto& father = Wavelet::InverseFather();
		const auto& mother = Wavelet::InverseMother();

		// How many wavelet kernels overlap this pair?
		const int halfSize = size / 2;

		for (int i = 0, j = 0; i < halfSize; i++, j += 2)
		{
			Real c0 = 0, c1 = 0;

			for (int b = B0; b <= B1; b++)
			{
				// Assume boundaries are reflected
				int bFather = i + b;
				int bMother = bFather;
				if (bFather < 0)
				{
					bFather = -bFather;
					bMother = -bMother - 1;
				}
				else if (bFather > halfSize - 1)
				{
					bFather = 1 + 2 * (halfSize - 1) - bFather;
					bMother = 2 * (halfSize - 1) - bMother;
				}

				const int k0 = -(2 * b) - K0;
				const int k1 = k0 + 1;
				if (k0 >= 0)
				{
					c0 += input[bFather] * father[k0] + input[bMother + halfSize] * mother[k0];
				}
				if (k1 < mother.size())
				{
					c1 += input[bFather] * father[k1] + input[bMother + halfSize] * mother[k1];
				}
			}

			output[j] = c0 * 2;
			output[j + 1] = c1 * 2;
		}
	}
};

