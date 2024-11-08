#pragma once

#include "Haar.h"
#include "D4.h"
#include "CDF53.h"
#include "CDF97.h"

namespace Flair
{
	/*
	*  Discrete wavelet transform parameterised by PrimaryWavelet mother/father pair.
	* */
	
	// 2D discrete wavelet transform made up of separate 1D transforms
	template<typename PrimaryWavelet>
	class DWT
	{
	public:
		using Real = typename PrimaryWavelet::kType;	

	protected:
		std::vector<Real> m_lineInputData;
		std::vector<Real> m_lineOutputData;

		int m_numPasses;
		int m_numPrimaryPasses;
		int m_blockSize;

	protected:
		void ForwardTransform(std::vector<Real>& inputData, const int passIdx, const int passSize)
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

		void InverseTransform(std::vector<Real>& inputData, const int passIdx, const int passSize)
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

		template<typename Functor>
		void TraversePrecinctInnerQuadrant(std::vector<Real>& inputData, const int quadrantSize, Functor onInner)
		{
			for (int y = 0; y < quadrantSize; ++y)
			{
				for (int x = 0; x < quadrantSize; ++x)
				{
					onInner(x, y);
				}
			}
		}

		template<typename Functor>
		void TraversePrecinctOuterQuadrants(std::vector<Real>& inputData, const int quadrantSize, Functor onOuter)
		{
			for (int y = 0; y < quadrantSize; ++y)
			{
				for (int x = 0; x < quadrantSize; ++x)
				{
					for (int v = 0; v < 2; ++v)
					{
						for (int u = 0; u < 2; ++u)
						{
							if (u != 0 || v != 0)
							{
								onOuter(x + quadrantSize * u, y + quadrantSize * v);
							}
						}
					}
				}
			}
		}

		void ValidateInput(std::vector<Real>& inputData)
		{
			AssertMsg(m_blockSize > 0, "DWT was not initialised with Prepare().");
			AssertFmt(inputData.size() >= sqr(m_blockSize), "Input data of size %zi is not large enough for block size of %i.", inputData.size(), m_blockSize);
		}

	public:
		DWT() : m_blockSize(0) {}

		void Prepare(const int blockSize)
		{
			// Allocate temporary storage that can be reused accross successive transforms
			m_blockSize = blockSize;
			m_lineInputData.resize(m_blockSize);
			m_lineOutputData.resize(m_blockSize);
			m_numPasses = int(std::log2(m_blockSize / Haar<Real>::GetMinIOSize())) + 1;
			m_numPrimaryPasses = int(std::floor(std::log2(m_blockSize / PrimaryWavelet::GetMinIOSize()))) + 1;
		}

		void Forward(std::vector<Real>& inputData)
		{
			ValidateInput(inputData);

			int passSize = m_blockSize;
			for (int passIdx = 0; passIdx < m_numPasses; passIdx++, passSize /= 2)
			{
				ForwardTransform(inputData, passIdx, passSize);	
			}
		}

		void Inverse(std::vector<Real>& inputData)
		{
			ValidateInput(inputData);
			
			int passSize = m_blockSize >> (m_numPasses - 1);
			for (int passIdx = m_numPasses - 1; passIdx >= 0; passIdx--, passSize *= 2)
			{
				InverseTransform(inputData, passIdx, passSize);
			}
		}
	};
}