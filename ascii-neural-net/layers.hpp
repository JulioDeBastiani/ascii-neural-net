#pragma once

#include <iostream>
#include <fstream>
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

        inline int in_size() const
        {
            return _in_size;
        }

        inline int out_size() const
        {
            return _out_size;
        }

        virtual Status forward(const RowVector& prev) = 0;
        virtual const RowVector& output() = 0;
        virtual Status backprop(RowVector prox) = 0;
        virtual const RowVector& backprop_output() = 0;
        virtual Status update(const RowVector& prev, Scalar learning_rate) = 0;

        virtual void serialize(std::ofstream& stream) const = 0;
        virtual Status deserialize(std::ifstream& stream) = 0;

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

        Status forward(const RowVector& prev)
        {
            if (prev.rows() != _in_size || prev.cols() != 1)
                return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");
            
            _output_mat = prev;
        }

        const RowVector& output()
        {
            return _output_mat;
        }

        Status backprop(RowVector prox)
        {
            return Status::OK();
        }

        const RowVector& backprop_output()
        {
            return _output_mat;
        }

        Status update(const RowVector& prev, Scalar learning_rate)
        {
            return Status::OK();
        }

        void serialize(std::ofstream& stream) const;
        Status deserialize(std::ifstream& stream);

    private:
        RowVector _output_mat;
    };

    class Dense : public Layer
    {
    public:
        Dense(int in_size, int out_size):
            Layer(in_size, out_size)
        {
            _weights = Matrix::Random(out_size, in_size);
        }

        inline int in_size()
        {
            return _in_size;
        }

        inline int out_size()
        {
            return _out_size;
        }

        Status forward(const RowVector& prev);

        const RowVector& output()
        {
            return _forward_output;
        }

        Status backprop(RowVector prox);

        const RowVector& backprop_output()
        {
            return _error;
        }

        Status update(const RowVector& prev, Scalar learning_rate);

        void serialize(std::ofstream& stream) const;
        Status deserialize(std::ifstream& stream);

    private:
        Matrix _weights;
        Matrix _z;
        RowVector _forward_output;

        RowVector _delta;
        RowVector _error;
    };
}