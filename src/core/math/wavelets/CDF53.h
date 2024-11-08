#pragma once

#include "Wavelet.h"

// Cohen–Daubechies–Feauveau 5/3 windowed wavelet
template<typename Real>
class CDF53 : public Wavelet1D<CDF53<Real>, Real, -2, 3>
{
public:
	CDF53() = delete;

	using kType = Real;
	static constexpr int kSize = 6;

	inline static const std::array<kType, kSize>& ForwardFather()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0,
			0.25,
			0.5,
			0.25,
			0,
			0
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& ForwardMother()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0,
			-0.125,
			-0.25,
			0.75,
			-0.25,
			-0.125
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& InverseFather()
	{
		static const std::array<Real, kSize> coeffs =
		{
			-0.125,
			0.25,
			0.75,
			0.25,
			-0.125,
			0
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& InverseMother()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0,
			0,
			-0.25,
			0.5,
			-0.25,
			0
		};
		return coeffs;
	}
};

