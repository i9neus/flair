#pragma once

#include "core/math/wavelets/DWT.h"

namespace Flair
{
	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename Real>
	class LiftingDWT : public DWT<Real>
	{
		using Base = DWT<Real>;

	protected:
		virtual void ForwardTransform(std::vector<Real>& inputData, const int passIdx, const int passSize) override final
		{
			for (int dimension = 0; dimension < 2; dimension++)
			{
				const int dx = (dimension + 1) % 2;
				const int dy = dimension;
				for (int line = 0; line < passSize; line++)
				{
					// Copy the data to the line buffer
					for (int element = 0; element < passSize; element++)
					{
						m_lineInputData[element] = inputData[(line * dx + element * dy) * m_blockSize + (line * dy + element * dx)];
					}

					if (passIdx < m_numPrimaryPasses)
					{
						// If the buffer width is large enough support the wideband wavelet, transform it now. 
						PrimaryWavelet::Forward(m_lineInputData, m_lineOutputData, passSize);
					}
					else
					{
						// Otherwise, fall back to using Haar to avoid needing to handle the buffer wrap-around
						Haar<Real>::Forward(m_lineInputData, m_lineOutputData, passSize);
					}

					// Copy the transformed data from the line buffer to the output
					for (int element = 0; element < passSize; element++)
					{
						inputData[(line * dx + element * dy) * m_blockSize + (line * dy + element * dx)] = m_lineOutputData[element];
					}
				}
			}
		}

		virtual void InverseTransform(std::vector<Real>& inputData, const int passIdx, const int passSize) override final			
		{
			for (int dimension = 0; dimension < 2; dimension++)
			{
				const int dx = dimension;
				const int dy = (dimension + 1) % 2;
				for (int line = 0; line < passSize; line++)
				{
					for (int element = 0; element < passSize; element++)
					{
						m_lineInputData[element] = inputData[(line * dx + element * dy) * m_blockSize + (line * dy + element * dx)];
					}

					if (passIdx < m_numPrimaryPasses)
					{
						PrimaryWavelet::Inverse(m_lineInputData, m_lineOutputData, passSize);
					}
					else
					{
						Haar<Real>::Inverse(m_lineInputData, m_lineOutputData, passSize);
					}

					for (int element = 0; element < passSize; element++)
					{
						inputData[(line * dx + element * dy) * m_blockSize + (line * dy + element * dx)] = m_lineOutputData[element];
					}
				}
			}
		}

	public:
		LiftingDWT(const int blockSize) : Base(blockSize)
		{
		
		}
		
	};
}