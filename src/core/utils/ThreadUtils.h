#pragma once

#if !defined(FLAIR_DISABLE_MULTITHREADING)
#include <thread>
#endif
#include <vector>
#include <functional>

namespace Flair
{
	template<typename Ctx>
	class Threaded
	{
		class Iterator
		{
			typename std::vector<Ctx>::iterator m_it;

		private:
			Iterator(typename std::vector<Ctx>::iterator& it) : m_it(it) {}

		public:
			inline Iterator& operator++() { ++m_it; return *this; }
			inline Iterator& operator--() { --m_it; return *this; }
			inline bool operator!=(const Iterator& other) const { return m_it != other.m_it; }
			inline Ctx& operator*() { return *m_it; }
			inline Ctx* operator->() { return &*m_it; }
		};

	public:
		using Functor = std::function<void(Ctx&, int, int)>;

	public:
		Threaded(const int maxThreads = 16)
		{
#if !defined(FLAIR_DISABLE_MULTITHREADING)
			int numThreads = std::max(1, int(std::thread::hardware_concurrency()));
			if (maxThreads > 0)
			{
				numThreads = std::min(maxThreads, numThreads);
			}
			m_ctxs.resize(numThreads);
#else
			m_ctxs.resize(1);
#endif
		}

		void Initialise(Functor onInit)
		{
			for (int i = 0; i < m_ctxs.size(); ++i)
			{
				onInit(m_ctxs[i], i, m_ctxs.size());
			}
		}

#if !defined(FLAIR_DISABLE_MULTITHREADING)

		void Run(Functor onExecute)
		{			
			for (int i = 0; i < m_ctxs.size(); ++i)
			{
				m_threads.emplace_back(&Threaded<Ctx>::RunThread, this, onExecute, std::ref(m_ctxs[i]), i);
			}

			// Wait for all the workers to finish
			for (int i = 0; i < m_ctxs.size(); ++i) { m_threads[i].join(); }
		}
#else
		void RunSerial(Functor onExecute)
		{
			for (int i = 0; i < m_ctxs.size(); ++i)
			{
				RunThread(onExecute, m_ctxs[i], 0);
			}
		}
#endif

		// Iterators
		inline Iterator begin() { return Iterator(m_ctxs.begin()); }
		inline Iterator end() { return Iterator(m_ctxs.end()); }

		std::vector<Ctx>& GetContexts() { return m_ctxs; }

	private:
		void RunThread(Functor onExecute, Ctx& ctx, int threadIdx)
		{
			onExecute(ctx, threadIdx, m_ctxs.size());
		}

	private:
		std::vector<Ctx> m_ctxs;
#if !defined(FLAIR_DISABLE_MULTITHREADING)
		std::vector<std::thread> m_threads;
#endif
	};
}