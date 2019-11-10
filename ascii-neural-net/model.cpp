#include <ascii-neural-net/model.hpp>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

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
        return _save_checkpoint(checkpoint_folder, ".checkpoint");
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

    const RowVector& Model::predict(const RowVector& input)
    {
        int total_layers = _layers.size();

        auto status = _forward(input);
        return _layers[total_layers]->output();
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

            eval(data, false);

            std::stringstream ss;
            ss << ".checkpoint." << std::setw(3) << std::setfill('0') << epoch;

            _save_checkpoint(checkpoints_folder, ss.str());
            _save_checkpoint(checkpoints_folder, ".checkpoint");

            std::cout << "saved checkpoint" << std::endl;
        }

        return Status::OK();
    }

    int argmax(const RowVector& vec)
    {
        int rows = vec.rows();
        int max_index = 0;
        Scalar  max_val = vec(0, 0);

        for (int i = 1; i < rows; i++)
        {
            Scalar val = vec(i, 0);

            if (val > max_val)
            {
                max_val = val;
                max_index = i;
            }
        }

        return max_index;
    }

    void Model::eval(Dataset& data, bool verbose)
    {
        double correct = 0;
        double incorrect = 0;
        
        while (!data.epoch_end())
        {
            auto d = data.next();
            auto output = predict(d->input());
            auto predicted = argmax(output);
            auto truth = argmax(d->expected_output());

            if (verbose)
                std::cout << "expected \"" << truth << "\", predicted: \"" << predicted << "\"\n";

            if (predicted == truth)
                correct += 1;
            else
                incorrect += 1;
        }

        double precision = (correct / (correct + incorrect)) * 100;
        std::cout << "precision: " << precision << std::endl;
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

    Status Model::_save_checkpoint(std::string checkpoint_folder, std::string extension)
    {
        std::ofstream file;
        file.open(checkpoint_folder + "/" + _name + extension, std::ios_base::out);

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
}