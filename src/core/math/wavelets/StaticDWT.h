#pragma once

#include "DWT.h"
#include "Haar.h"
#include "D4.h"
#include "CDF53.h"
#include "CDF97.h"

namespace Flair
{
	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename PrimaryWavelet>
	class StaticDWT : public DWT<typename PrimaryWavelet::kType>
	{
	private:
		using Base = DWT<typename PrimaryWavelet::kType>;

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
				PrimaryWavelet::Forward(m_lineInputData, m_lineOutputData, passSize);
			}
			else
			{
				// Otherwise, fall back to using Haar to avoid needing to handle the buffer wrap-around
				Haar<Real>::Forward(m_lineInputData, m_lineOutputData, passSize);
			}
		}

		virtual void InverseTransform1D(const int passIdx, const int passSize) override final
		{
			if (passIdx < m_numPrimaryPasses)
			{
				PrimaryWavelet::Inverse(m_lineInputData, m_lineOutputData, passSize);
			}
			else
			{
				Haar<Real>::Inverse(m_lineInputData, m_lineOutputData, passSize);
			}
		}

	public:
		StaticDWT(const int blockSize) : Base(blockSize)
		{
			Base::m_numPasses = int(std::log2(m_blockSize / Haar<Real>::GetMinIOSize())) + 1;
			m_numPrimaryPasses = int(std::floor(std::log2(m_blockSize / PrimaryWavelet::GetMinIOSize()))) + 1;
		}
	};
}