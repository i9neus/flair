#pragma once

#include <thread>
#include <atomic>
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
		Threaded(const int numThreads, const int pollInterval = 10) : m_ctxs(numThreads), m_pollInterval(std::max(1, pollInterval))
		{	}

		template<typename Lambda>
		void Initialise(Lambda onInit)
		{
			for (int i = 0; i < m_ctxs.size(); ++i)
			{
				onInit(m_ctxs[i], i, m_ctxs.size());
			}
		}

		template<typename Lambda>
		void Run(Lambda onExecute)
		{
			m_activeCount.store(m_ctxs.size());
			for (int i = 0; i < m_ctxs.size(); ++i)
			{
				std::function<void(Ctx&)> fn = onExecute;
				m_threads.emplace_back(&Threaded<Ctx>::RunImpl, this, onExecute, std::ref(m_ctxs[i]));
				m_threads.back().detach();
			}

			// Wait for everything to finish
			while (m_activeCount.load() > 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(m_pollInterval));
			}
		}

		template<typename Lambda>
		void RunSerial(Lambda onExecute)
		{
			m_activeCount.store(m_ctxs.size());
			for (int i = 0; i < m_ctxs.size(); ++i)
			{
				RunImpl(onExecute, m_ctxs[i]);
			}
		}

		// Iterators
		inline Iterator begin() { return Iterator(m_ctxs.begin()); }
		inline Iterator end() { return Iterator(m_ctxs.end()); }

		std::vector<Ctx>& GetContexts() { return m_ctxs; }

	private:
		void RunImpl(std::function<void(Ctx&)> onExecute, Ctx& ctx)
		{
			onExecute(ctx);
			m_activeCount.fetch_sub(1);
		}

	private:
		std::vector<Ctx> m_ctxs;
		std::vector<std::thread> m_threads;
		std::atomic<int> m_activeCount;
		int m_pollInterval;
	};
}