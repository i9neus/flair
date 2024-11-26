#pragma once

#include "Wavelet1.h"

// Cohen–Daubechies–Feauveau 9/7 windowed wavelet
template<typename Real>
class CDF97 : public Wavelet1<CDF97<Real>, Real, -4, 5>
{
public:
	CDF97() = delete;

	using kType = Real;
	static constexpr int kSize = 10;

	inline static const std::array<kType, kSize>& ForwardFather()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0.026748757411,
			-0.016864118443,
			-0.078223266529,
			0.266864118443,
			0.602949018236,
			0.266864118443,
			-0.078223266529,
			-0.016864118443,
			0.026748757411,
			0
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& ForwardMother()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0,
			0,
			0.091271763114,
			-0.057543526229,
			-0.591271763114,
			1.11508705,
			-0.591271763114,
			-0.057543526229,
			0.091271763114,
			0
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& InverseFather()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0,
			-0.091271763114 * 0.5,
			-0.057543526229 * 0.5,
			0.591271763114 * 0.5,
			1.11508705 * 0.5,
			0.591271763114 * 0.5,
			-0.057543526229 * 0.5,
			-0.091271763114 * 0.5,
			0,
			0
		};
		return coeffs;
	}

	inline static const std::array<kType, kSize>& InverseMother()
	{
		static const std::array<Real, kSize> coeffs =
		{
			0,
			0.026748757411 * 0.5,
			0.016864118443 * 0.5,
			-0.078223266529 * 0.5,
			-0.266864118443 * 0.5,
			0.602949018236 * 0.5,
			-0.266864118443 * 0.5,
			-0.078223266529 * 0.5,
			0.016864118443 * 0.5,
			0.026748757411 * 0.5
		};
		return coeffs;
	}
};