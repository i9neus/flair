#pragma once

#include "DWT1.h"

namespace Flair
{
	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename Real>
	class LiftingDWT1 : public DWT1<Real>
	{
	private:
		using Base = DWT1<Real>;

	protected:
		virtual void ForwardTransform1D(const int passIdx, const int passSize) override final
		{
			const int halfSize = passSize / 2;
			for (int i = 0, j = 0; i < passSize; i += 2, j++)
			{
				m_lineOutputData[j] = m_lineInputData[i];
				m_lineOutputData[j + halfSize] = m_lineInputData[i + 1] - m_lineInputData[i];
			}
		}

		virtual void InverseTransform1D(const int passIdx, const int passSize) override final
		{
			const int halfSize = passSize / 2;
			for (int i = 0, j = 0; i < halfSize; i++, j += 2)
			{
				m_lineOutputData[j] = m_lineInputData[i];
				m_lineOutputData[j + 1] = m_lineInputData[i] + m_lineInputData[i + halfSize];
			}
		}

	public:
		LiftingDWT1(const int blockSize) : Base(blockSize) {}
	};
}