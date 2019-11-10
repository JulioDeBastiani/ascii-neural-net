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

        void serialize(std::ofstream& stream) const
        {
            stream << "no weights\n";
        }

        Status deserialize(std::ifstream& stream)
        {
            std::string line;

            if (!getline(stream, line))
                return Status::ERROR(Status::error_codes::DESERIALIZATION_ERROR, "could not read");

            if (line != "no weights")
                return Status::ERROR(Status::error_codes::DESERIALIZATION_ERROR, "");

            return Status::OK();
        }

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

        Status forward(const RowVector& prev)
        {
            // calculate neuron activations
            _z.resize(_out_size, 1);
            _z.noalias() = _weights * prev;

            // TODO bias would be nice

            // apply softmax activation function
            // TODO support other activation functions
            _forward_output.resize(_out_size, 1);
            _forward_output.array() = Scalar(1) / (Scalar(1) + (-_z.array()).exp());
        }

        const RowVector& output()
        {
            return _forward_output;
        }

        Status backprop(RowVector prox)
        {
            if (prox.rows() != _out_size)
                return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");
            
            // delta for this layer
            _delta.resize(_out_size, 1);
            _delta.noalias() = _forward_output.cwiseProduct(RowVector::Ones(_out_size) - _forward_output).cwiseProduct(prox);

            // error factor for the previous layer
            _error.resize(_in_size, 1);
            _error.noalias() = (_delta.transpose() * _weights).transpose();

            return Status::OK();
        }

        const RowVector& backprop_output()
        {
            return _error;
        }

        Status update(const RowVector& prev, Scalar learning_rate)
        {
            if (prev.rows() != _in_size)
                return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");

            if (_error.rows() != _out_size)
                return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");
            
            _weights.noalias() += learning_rate * (prev * _error);
            return Status::OK();
        }

        void serialize(std::ofstream& stream) const
        {
            stream << _weights << "\n";
        }

        Status deserialize(std::ifstream& stream)
        {
            std::string line;
            
            for (int row = 0; row < _in_size; row++)
            {
                for (int col = 0; col < _out_size; col++)
                {
                    Scalar val;
                    stream >> val;
                    _weights(row, col) = val;
                }
            }
        }

    private:
        Matrix _weights;
        Matrix _z;
        RowVector _forward_output;

        RowVector _delta;
        RowVector _error;
    };
}