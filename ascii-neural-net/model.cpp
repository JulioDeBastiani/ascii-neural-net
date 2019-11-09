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
                boost::split(line, tokens, [](char c){ return c == ' '; });

                if (tokens[0] == "Input")
                {
                    prev = std::stoi(tokens[1]);
                    _layers.push_back(std::make_unique<Input>(prev));
                    continue;
                }

                if (tokens[0] == "Dense")
                {
                    if (tokens[2] == "Sigmoid")
                    {
                        int c = std::stoi(tokens[1]);
                        _layers.push_back(std::make_unique<Dense<Sigmoid>>(prev, c));
                        prev = c;
                    }

                    if (tokens[2] == "ReLU")
                    {
                        int c = std::stoi(tokens[1]);
                        _layers.push_back(std::make_unique<Dense<ReLU>>(prev, c));
                        prev = c;
                    }
                }
            }
        }

    //     Status load_checkpoint(std::string checkpoint_folder);

    //     // TODO fix inputs & outputs to tensors or mats
    //     std::vector<float> predict(std::vector<int> input);
    //     Status fit(std::vector<std::vector<int>> train_date, std::vector<std::vector<float>> train_labels, float learning_rate, int epochs, std::string checkpoints_folder);
        
    // private:
    //     std::string _name;
    
    //     std::vector<Layer> _layers;

    //     float _calculate_loss(std::vector<float> predicted_labels, std::vector<float> actual_labels);
    //     Status update_weights(float loss, float learning_rate);
}