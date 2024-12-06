#pragma once

#include "DWT1.h"
#include "Haar1.h"
#include "D4.h"
#include "CDF53.h"
#include "CDF97.h"

namespace Flair
{
	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename PrimaryWavelet>
	class StaticDWT1 : public DWT1<typename PrimaryWavelet::kType>
	{
	private:
		using Base = DWT1<typename PrimaryWavelet::kType>;

	protected:
		int m_numPrimaryPasses;

	public:
		using Real = typename PrimaryWavelet::kType;	

	protected:		 
		virtual void ForwardTransform1D(const int passIdx, const int passSize) override final
		{
			if (passIdx < m_numPrimaryPasses)
			{
				// If the buffer width is large enough support the wideband wavelet, transform it now. 
				PrimaryWavelet::Forward(Base::m_lineInputData, Base::m_lineOutputData, passSize);
			}
			else
			{
				// Otherwise, fall back to using Haar to avoid needing to handle the buffer wrap-around
				Haar1<Real>::Forward(Base::m_lineInputData, Base::m_lineOutputData, passSize);
			}
		}

		virtual void InverseTransform1D(const int passIdx, const int passSize) override final
		{
			if (passIdx < m_numPrimaryPasses)
			{
				PrimaryWavelet::Inverse(Base::m_lineInputData, Base::m_lineOutputData, passSize);
			}
			else
			{
				Haar1<Real>::Inverse(Base::m_lineInputData, Base::m_lineOutputData, passSize);
			}
		}

	public:
		StaticDWT1(const int blockSize) : Base(blockSize)
		{
			Base::m_numPasses = int(std::log2(Base::m_blockSize / Haar1<Real>::GetMinIOSize())) + 1;
			m_numPrimaryPasses = int(std::floor(std::log2(Base::m_blockSize / PrimaryWavelet::GetMinIOSize()))) + 1;
		}
	};
}