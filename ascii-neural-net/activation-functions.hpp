#pragma once

#include <ascii-neural-net/types.hpp>

namespace ann
{
    class Sigmoid
    {
    public:
        static inline void apply(const Matrix& input, Matrix& output)
        {
            // TODO verify function/write tests
            output.array() = Scalar(1) / (Scalar(1) + (-input.array()).exp());
        }
    };

    class ReLU
    {
    public:
        static inline void apply(const Matrix& input, Matrix& output)
        {
            // TODO verify function/write tests
            output.array() = input.cwiseMax(Scalar(0));
        }
    };
}