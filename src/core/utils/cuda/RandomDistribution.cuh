#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include <random>

namespace Flair
{
    class RandomDistribution
    {
    protected:
        std::mt19937 m_mt;
    
    protected:
        RandomDistribution(const uint32_t seed) : m_mt(seed) {}
        RandomDistribution() = default;

    public:
        virtual float operator()() = 0;
    };
    
    // Continuous uniform distribution in the range [lower, upper)
    class UniformDistribution : public RandomDistribution
    {
    private:        
        std::uniform_real_distribution<float> m_rng;
        float m_lower, m_upper;

    public:
        UniformDistribution(const float lower, const float upper, const uint32_t seed = 0) : RandomDistribution(seed), m_lower(lower), m_upper(upper) {}
        virtual float operator()() override final { return mix(m_lower, m_upper, m_rng(m_mt)); }
    };

    // Normal distribution with the option of mapping to log
    template<bool LogNormal>
    class NormalDistributionImpl : public RandomDistribution
    {
    private:
        std::normal_distribution<float> m_rng;

    public:
        NormalDistributionImpl(const float mean, const float sigma, const uint32_t seed = 0) : RandomDistribution(seed), m_rng(mean, sigma) {}
        virtual float operator()() override final 
        { 
            return LogNormal ? std::exp(m_rng(m_mt)) : m_rng(m_mt);
        }
    };
    using NormalDistribution = NormalDistributionImpl<false>;
    using LogNormalDistribution = NormalDistributionImpl<true>;        

    // Always returns 1
    class Ones : public RandomDistribution
    {
    public:
        Ones() = default;
        virtual float operator()() override final { return 1.0f; }
    };
}
