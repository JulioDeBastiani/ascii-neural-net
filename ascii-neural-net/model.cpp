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

    //     // TODO fix inputs & outputs to tensors or mats
    //     std::vector<float> predict(std::vector<int> input);
    //     Status fit(std::vector<std::vector<int>> train_date, std::vector<std::vector<float>> train_labels, float learning_rate, int epochs, std::string checkpoints_folder);
        
    // private:
    //     std::string _name;
    
    //     std::vector<Layer> _layers;

    //     float _calculate_loss(std::vector<float> predicted_labels, std::vector<float> actual_labels);
    //     Status update_weights(float loss, float learning_rate);
}