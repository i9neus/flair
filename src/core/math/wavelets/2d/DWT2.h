#pragma once

#include "core/math/MathUtils.h"
#include <vector>

namespace Flair
{
	/*
	*  2D discrete wavelet transform parameterised by PrimaryWavelet mother/father pair.
	* */
	template<typename Real>
	class DWT2
	{
	protected:
		std::vector<Real>	m_swap; 
		int					m_size;
		int					m_numPasses;

	public:
		void Forward(std::vector<Real>& inputData)
		{
			ValidateInput(inputData);

			int passSize = m_size;
			for (int passIdx = 0; passIdx < m_numPasses; passIdx++, passSize >>= 1)
			{
				ForwardTransform(inputData, passIdx, passSize); 
			}
		}

		void Inverse(std::vector<Real>& inputData)
		{
			ValidateInput(inputData);

			int passSize = m_size >> (m_numPasses - 1);
			for (int passIdx = m_numPasses - 1; passIdx >= 0; passIdx--, passSize <<= 1)
			{
				InverseTransform(inputData, passIdx, passSize);
			}
		}

	protected:
		DWT2(const int size, const int maxPasses = std::numeric_limits<int>::max())
		{
			// Allocate temporary storage that can be reused accross successive transforms
			m_size = size;
			m_numPasses = std::min(maxPasses, int(std::round(std::log2(m_size / 2))) + 1);
			m_swap.resize(m_size * m_size);
		}

		virtual void ForwardTransform(std::vector<Real>& srcBuffer, const int passIdx, const int passSize) = 0;
		virtual void InverseTransform(std::vector<Real>& srcBuffer, const int passIdx, const int passSize) = 0;

		void CopyPassFromSwap(std::vector<Real>& srcBuffer, const int passSize)
		{
			for (int y = 0; y < passSize; ++y)
			{
				std::memcpy(&srcBuffer[y * m_size], &m_swap[y * m_size], sizeof(Real) * passSize);
			}
		}

		void ValidateInput(std::vector<Real>& inputData)
		{
			AssertMsg(m_size > 0, "DWT1 was not initialised with Prepare().");
			AssertFmt(inputData.size() >= m_swap.size(), "Input data of size %zi is not large enough for block size of %i.", inputData.size(), m_size);
		}
	};
}