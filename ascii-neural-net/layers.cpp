#include <ascii-neural-net/layers.hpp>

namespace ann
{
    void Input::serialize(std::ofstream& stream) const
    {
        stream << "no weights\n";
    }

    Status Input::deserialize(std::ifstream& stream)
    {
        std::string line;

        if (!getline(stream, line))
            return Status::ERROR(Status::error_codes::DESERIALIZATION_ERROR, "could not read");

        if (line != "no weights")
            return Status::ERROR(Status::error_codes::DESERIALIZATION_ERROR, "");

        return Status::OK();
    }

    Status Dense::forward(const RowVector& prev)
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

    Status Dense::backprop(RowVector prox)
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

    Status Dense::update(const RowVector& prev, Scalar learning_rate)
    {
        if (prev.rows() != _in_size)
            return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");

        if (_delta.rows() != _out_size)
            return Status::ERROR(Status::error_codes::INCOMPATIBLE_SIZES, "");
            
        _weights.noalias() += learning_rate * (_delta * prev.transpose());
        return Status::OK();
    }

    void Dense::serialize(std::ofstream& stream) const
    {
        stream << _weights << "\n";
    }

    Status Dense::deserialize(std::ifstream& stream)
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
}