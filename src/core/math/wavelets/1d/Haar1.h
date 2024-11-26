#pragma once

#include "Wavelet1.h"

// Haar wavelet
template<typename Real>
class Haar1 : public Wavelet1<Haar1<Real>, Real, 0, 1>
{
public:
	Haar1() = delete;

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

