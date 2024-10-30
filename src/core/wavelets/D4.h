#pragma once

#include "Wavelet.h"

// Daubechies 4-tap wavelet
template<typename Real>
class D4 : public Wavelet1D<D4<Real>, Real, -1, 2>
{
public:
	D4() = delete;

	using kType = typename Real;
	static constexpr int kSize = 4;

	inline static const std::array<kType, kSize>& ForwardFather()
	{
		static const std::array<Real, kSize> coeffs =
		{
			(1 + std::sqrt(3.0)) / (4 * 2),
			(3 + std::sqrt(3.0)) / (4 * 2),
			(3 - std::sqrt(3.0)) / (4 * 2),
			(1 - std::sqrt(3.0)) / (4 * 2)
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& ForwardMother()
	{
		static const std::array<Real, kSize> coeffs =
		{
			(1 - std::sqrt(3.0)) / (4 * 2),
			-(3 - std::sqrt(3.0)) / (4 * 2),
			(3 + std::sqrt(3.0)) / (4 * 2),
			-(1 + std::sqrt(3.0)) / (4 * 2)
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& InverseFather() { return ForwardFather(); }
	inline static const std::array<kType, kSize>& InverseMother() { return ForwardMother(); }
};