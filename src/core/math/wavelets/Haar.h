#pragma once

#include "Wavelet.h"

// Haar wavelet
template<typename Real>
class Haar : public Wavelet1D<Haar<Real>, Real, 0, 1>
{
public:
	Haar() = delete;

	using kType = Real;
	static constexpr int kSize = 2;

	inline static const std::array<kType, kSize>& ForwardFather()
	{
		static const std::array<Real, 2> coeffs = { 0.5, 0.5 };
		return coeffs;
	}

	inline static const std::array<kType, kSize>& ForwardMother()
	{
		static const std::array<Real, 2> coeffs = { 0.5, -0.5 };
		return coeffs;
	}

	inline static const std::array<kType, kSize>& InverseFather() { return ForwardFather(); }
	inline static const std::array<kType, kSize>& InverseMother() { return ForwardMother(); }
};

