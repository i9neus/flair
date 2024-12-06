#pragma once

#include "DWT2.h"

namespace Flair
{
	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename Real>
	class StaticDWT2 : public DWT2<Real>
	{
	private:
		using Base = DWT2<Real>;

	protected:
		virtual void ForwardTransform(std::vector<Real>& srcBuffer, const int passIdx, const int passSize) override final
		{
			const int halfSize = passSize / 2;
			for (int y = 0; y < passSize; y += 2)
			{
				for (int x = 0; x < passSize; x += 2)
				{
					const Real f00 = srcBuffer[y * Base::m_size + x];
					const Real f10 = srcBuffer[y * Base::m_size + (x + 1)];
					const Real f01 = srcBuffer[(y + 1) * Base::m_size + x];
					const Real f11 = srcBuffer[(y + 1) * Base::m_size + (x + 1)];

					const Real h00 = (f00 + f10 + f01 + f11) * 0.25;
					const Real norm = std::max(1.0f, h00);

					Base::m_swap[(y >> 1) * Base::m_size + (x >> 1)]							= h00;
					Base::m_swap[(y >> 1) * Base::m_size + (x >> 1) + halfSize]					= (-f00 + f10 - f01 + f11) * 0.25 / norm;	// h10
					Base::m_swap[((y >> 1) + halfSize) * Base::m_size + (x >> 1)]				= (-f00 - f10 + f01 + f11) * 0.25 / norm;	// h01
					Base::m_swap[((y >> 1) + halfSize) * Base::m_size + (x >> 1) + halfSize]	= (f00 - f10 - f01 + f11) * 0.25 / norm;	// h11
				}
			}

			Base::CopyPassFromSwap(srcBuffer, passSize);
		}

		virtual void InverseTransform(std::vector<Real>& srcBuffer, const int passIdx, const int passSize) override final
		{
			/* 
				2-dimensional Haar mother wavelet:
			
				h10 = -1  1    h01 = -1 -1     h11 = 1  -1
					  -1  1           1  1          -1   1 
			*/
			
			const int halfSize = passSize / 2;
			for (int y = 0; y < passSize; y += 2)
			{
				for (int x = 0; x < passSize; x += 2)
				{
					const Real h00 = srcBuffer[(y >> 1) * Base::m_size + (x >> 1)];
					const Real h10 = srcBuffer[(y >> 1) * Base::m_size + (x >> 1) + halfSize];
					const Real h01 = srcBuffer[((y >> 1) + halfSize) * Base::m_size + (x >> 1)];
					const Real h11 = srcBuffer[((y >> 1) + halfSize) * Base::m_size + (x >> 1) + halfSize];

					const Real norm = std::max(1.0f, h00);

					Base::m_swap[y * Base::m_size + x]				= h00 + (-h10 - h01 + h11) * norm;	// f00
					Base::m_swap[y * Base::m_size + (x + 1)]		= h00 + (h10 - h01 - h11) * norm;	// f10
					Base::m_swap[(y + 1) * Base::m_size + x]		= h00 + (-h10 + h01 - h11) * norm;	// f01
					Base::m_swap[(y + 1) * Base::m_size + (x + 1)]	= h00 + (h10 + h01 + h11) * norm;	// f11
				}
			}

			Base::CopyPassFromSwap(srcBuffer, passSize);
		}

	public:
		StaticDWT2(const int size, const int maxPasses = std::numeric_limits<int>::max()) : Base(size, maxPasses) {}
	};
}