#include <ascii-neural-net/model.hpp>

#include <iostream>
#include <fstream>

#include <boost/algorithm/string.hpp>

#include <ascii-neural-net/activation-functions.hpp>

namespace ann
{
    Model::Model(std::string name):
        _name(name)
    {
    }

    Status Model::load_model(std::string filename)
    {
        std::ifstream model_file;
        model_file.open(filename);

        if (!model_file.is_open())
        {
            std::cout << "could not open file \"" << filename << "\"" << std::endl;
            return Status::ERROR(Status::error_codes::PLACEHOLDER, "");
        }

        _layers.clear();

        std::string line;

        int prev = 0;

        while (getline(model_file, line))
        {
            std::vector<std::string> tokens;
            boost::split(tokens, line, [](char c){ return c == ' '; });

            if (tokens[0] == "Input")
            {
                prev = std::stoi(tokens[1]);
                _layers.push_back(std::make_unique<Input>(prev));
                std::cout << "Input: " << prev << std::endl;
                continue;
            }

            if (tokens[0] == "Dense")
            {
                int c = std::stoi(tokens[1]);
                _layers.push_back(std::make_unique<Dense>(prev, c));
                std::cout << "Dense: " << c << std::endl;
                prev = c;
                continue;
            }
        }

        return Status::OK();
    }

    Status Model::save_checkpoint(std::string checkpoint_folder)
    {
        std::ofstream file;
        file.open(checkpoint_folder + "/" + _name + ".checkpoint", std::ios_base::out);

        if (!file.is_open())
        {
            std::cout << "could not open file" << std::endl;
            return Status::ERROR(Status::error_codes::PLACEHOLDER, "could not open file");
        }

        for (auto& layer: _layers)
        {
            layer->serialize(file);
        }

        return Status::OK();
    }

    Status Model::load_checkpoint(std::string checkpoint_folder)
    {
        std::ifstream checkpoint;
        checkpoint.open(checkpoint_folder + "/" + _name + ".checkpoint");

        if (!checkpoint.is_open())
        {
            std::cout << "could not open file \"" << checkpoint_folder << "\"" << std::endl;
            return Status::ERROR(Status::error_codes::PLACEHOLDER, "");
        }

        for (auto& layer: _layers)
        {
            layer->deserialize(checkpoint);
        }

        return Status::OK();
    }

    Status Model::fit(Dataset& data, Scalar learning_rate, int epochs, std::string checkpoints_folder)
    {
        std::cout << "training model \"" << _name << "\"" << std::endl;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            std::cout << "epoch " << epoch << "/" << epochs << std::endl;

            while (!data.epoch_end())
            {
                auto d = data.next();

                auto status = _forward(d->input());

                if (status.err())
                {
                    std::cout << "Training error on forward" << std::endl;
                    return status;
                }

                status = _backprop(d->expected_output());

                if (status.err())
                {
                    std::cout << "Training error on backprop" << std::endl;
                    return status;
                }

                status = _update(learning_rate);


                if (status.err())
                {
                    std::cout << "Training error on update" << std::endl;
                    return status;
                }
            }

            save_checkpoint(checkpoints_folder);
        }

        return Status::OK();
    }

    Status Model::_forward(const RowVector& input)
    {
        int total_layers = _layers.size();
            
        if (total_layers == 0)
        {
            std::cout << "Uninitialized network" << std::endl;
            return Status::ERROR(Status::error_codes::PLACEHOLDER, "Uninitialized network");
        }

        if (input.rows() != _layers[0]->in_size())
        {
            std::cout << "Invalid input shape" << std::endl;
            return Status::ERROR(Status::error_codes::PLACEHOLDER, "Invalid input shape");
        }

        _layers[0]->forward(input);

        for (int i = 1; i < total_layers; i++)
        {
            auto status = _layers[i]->forward(_layers[i - 1]->output());

            if (status.err())
            {
                std::cout << "forwarding error" << std::endl;
                return status;
            }
        }

        _output = _layers[total_layers - 1]->output();

        return Status::OK();
    }

    Status Model::_backprop(const RowVector& expected_output)
    {
        RowVector error = expected_output - _output;
        int total_layers = _layers.size();

        for (int i = total_layers - 1; 1 >= 0; i--)
        {
            auto& layer = _layers[i];

            auto status = layer->backprop(error);

            if (status.err())
            {
                std::cout << "backpropagation error" << std::endl;
                return status;
            }

            error = layer->backprop_output();
        }

        return Status::OK();
    }

    Status Model::_update(Scalar learning_rate)
    {
        int total_layers = _layers.size();
            
        if (total_layers == 0)
        {
            std::cout << "Uninitialized network" << std::endl;
            return Status::ERROR(Status::error_codes::PLACEHOLDER, "Uninitialized network");
        }

        RowVector output = _layers[0]->output();

        for (int i = 1; i < total_layers; i++)
        {
            auto status = _layers[i]->update(output, learning_rate);

            if (status.err())
            {
                std::cout << "update error" << std::endl;
                return status;
            }

            output = _layers[i]->output();
        }

        return Status::OK();
    }
}