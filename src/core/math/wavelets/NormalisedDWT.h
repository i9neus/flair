#pragma once

#include "DWT.h"

namespace Flair
{
	/*
	*  Discrete wavelet transform parameterised by PrimaryWavelet mother/father pair.
	*  The purpose of this transform is return wavelet 
	* */
	
	enum Flags : uint32_t { kDWTLogSpace = 1 };

	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename PrimaryWavelet>
	class NormalisedDWT : public DWT<PrimaryWavelet>
	{
	public:
		using Real = typename PrimaryWavelet::kType;
		static constexpr int kNormKernelSize = 0;		

	private:
		using Base = DWT<PrimaryWavelet>;
		uint32_t m_flags;

	public:
		NormalisedDWT() : 
			Base(),
			m_flags(0) {}

		NormalisedDWT(const int blockSize, const uint32_t flags) : NormalisedDWT()
		{
			Prepare(blockSize, flags);
		}

		void Prepare(const int blockSize, const uint32_t flags) 
		{
			m_flags = flags;
			Base::Prepare(blockSize);
		}

		void Forward(std::vector<Real>& inputData, std::vector<float>& passNorms)
		{
			Base::ValidateInput(inputData);

			//std::printf("Haar passes: %i\n", numPasses);
			//std::printf("CDF 9/7 passes: %i\n", numPrimaryPasses);

			passNorms.resize(Base::m_numPasses, 1.f);

			int passSize = Base::m_blockSize;
			for (int passIdx = 0; passIdx < Base::m_numPasses; passIdx++)
			{
				// Regular DWT forward pass
				Base::ForwardTransform(inputData, passIdx, passSize);
				passSize /= 2;

				// Apply a first pass of local normalisation by dividing each detail coefficient in the outer quadrants by its 
				// corresponding averaged coefficient in the inner quadrant. 
				float maxAbsVal = 1;
				auto localNormalise = [&, this](const int x, const int y)
				{
					float peak = 1.;
					for (int v = -kNormKernelSize; v <= kNormKernelSize; ++v)
					{
						for (int u = -kNormKernelSize; u <= kNormKernelSize; ++u)
						{
							if (x + u >= 0 && x + u < passSize && y + v >= 0 && y + v < passSize)
							{
								peak = std::max(peak, std::abs(inputData[(y + v) * Base::m_blockSize + (x + u)]));
							}
						}
					}

					for (int v = 0; v < 2; ++v)
					{
						for (int u = 0; u < 2; ++u)
						{
							if (u != 0 || v != 0)
							{
								auto& f = inputData[(y + v * passSize) * Base::m_blockSize + (x + u * passSize)];
								if (passIdx < Base::m_numPrimaryPasses)
								{
									f /= peak;
								}

								if (m_flags & kDWTLogSpace)
								{
									f = std::log(1. + std::abs(f)) * sign(f);
								}

								maxAbsVal = std::max(maxAbsVal, std::abs(f));
							}
						}
					}
				};
				Base::TraversePrecinctInnerQuadrant(inputData, passSize, localNormalise);

				// If the peak absolute coefficient is larger than 1, normalise the outer quadrants to the peak.
				// This will lose some precision during quantisation, however it's preferable to clamping or using a hard-coded norm
				if (maxAbsVal > 1.)
				{
					Base::TraversePrecinctOuterQuadrants(inputData, passSize, [&, this](const int x, const int y)
						{
							inputData[y * Base::m_blockSize + x] /= maxAbsVal;
						});
					passNorms[passIdx] = maxAbsVal;
				}

				//std::printf("DWT Pass %i: %f\n", passIdx, maxAbsVal);
			}
		}

		void Inverse(std::vector<Real>& inputData, const std::vector<float>& passNorms)
		{
			Base::ValidateInput(inputData);

			int passSize = Base::m_blockSize >> (Base::m_numPasses - 1);
			for (int passIdx = Base::m_numPasses - 1; passIdx >= 0; passIdx--, passSize *= 2)
			{
				// If the norm for this pass is greater than 1, scale the outer quadrant coefficients accordingly
				if (passNorms[passIdx] > 1.)
				{
					const float maxAbsVal = passNorms[passIdx];
					Base::TraversePrecinctOuterQuadrants(inputData, passSize / 2, [&](const int x, const int y)
						{
							inputData[y * Base::m_blockSize + x] *= maxAbsVal;
						});
				}

				// Denormalise the outer coefficients based on the newly-decoded inner coefficient
				auto localDenormalise = [&](const int x, const int y)
				{
					float peak = 1.;
					for (int v = -kNormKernelSize; v <= kNormKernelSize; ++v)
					{
						for (int u = -kNormKernelSize; u <= kNormKernelSize; ++u)
						{
							if (x + u >= 0 && x + u < passSize / 2 && y + v >= 0 && y + v < passSize / 2)
							{
								peak = std::max(peak, std::abs(inputData[(y + v) * Base::m_blockSize + (x + u)]));
							}
						}
					}

					for (int v = 0; v < 2; ++v)
					{
						for (int u = 0; u < 2; ++u)
						{
							if (u != 0 || v != 0)
							{
								float& f = inputData[(y + v * passSize / 2) * Base::m_blockSize + (x + u * passSize / 2)];

								if (m_flags & kDWTLogSpace)
								{
									f = (std::exp(std::abs(f)) - 1) * sign(f);
								}

								if (passIdx < Base::m_numPrimaryPasses)
								{
									f *= peak;
								}
							}
						}
					}
				};
				Base::TraversePrecinctInnerQuadrant(inputData, passSize / 2, localDenormalise);

				// Regular DWT inverse pass
				Base::InverseTransform(inputData, passIdx, passSize);
			}
		}
	};
}