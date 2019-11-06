#pragma once

#include <vector>

#include <ascii-neural-net/status.hpp>
#include <ascii-neural-net/types.hpp>

namespace ann
{
    class Layer
    {
    public:
        Layer(int in_size, int out_size):
            _in_size(in_size),
            _out_size(out_size)
        {
        }

        virtual ~Layer() = 0;

        inline int in_size()
        {
            return _in_size;
        }

        inline int out_size()
        {
            return _out_size;
        }

        virtual Status foreward(const Flat& prev) = 0;
        virtual const Flat& output() = 0;
        virtual Status backprop(Flat prox) = 0;
        virtual const Flat& backprop_output() = 0;
        virtual Status update(Scalar learning_rate) = 0;

    protected:
        int _in_size;
        int _out_size;
    };

    class Input : public Layer
    {
    public:
        Input(int size):
            Layer(size, size)
        {
            _output_mat.resize(size);
        }

        Status foreward(const Flat& prev)
        {
            if (prev.rows() != 1 || prev.cols() != _in_size)
                return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");
            
            _output_mat = prev;
        }

        const Flat& output()
        {
            return _output_mat;
        }

        Status backprop(Flat prox)
        {
            return Status::OK();
        }

        const Flat& backprop_output()
        {
            // TODO
        }

        Status update(Scalar learning_rate)
        {
            return Status::OK();
        }

    private:
        Flat _output_mat;
    };

    template<typename ActivationFn>
    class Dense : public Layer
    {
    public:
        Dense(int in_size, int out_size):
            Layer(in_size, out_size)
        {
            _weights.resize(in_size, out_size);
        }

        inline int in_size()
        {
            return _in_size;
        }

        inline int out_size()
        {
            return _out_size;
        }

        virtual Status foreward(const Flat& prev) = 0;
        virtual const Flat& output() = 0;
        virtual Status backprop(Flat prox) = 0;
        virtual const Flat& backprop_output() = 0;
        virtual Status update(Scalar learning_rate) = 0;

    private:
        Matrix _weights;
    };
}