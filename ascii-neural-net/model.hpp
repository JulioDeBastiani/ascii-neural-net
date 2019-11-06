#pragma once

#include <string>
#include <vector>

#include <ascii-neural-net/layers.hpp>
#include <ascii-neural-net/status.hpp>

namespace ann
{
    // TODO impl class
    // TODO rule of five
    // TODO fix tensor types
    class Model
    {
    public:
        Model(std::string name);

        Status load_model(std::string filename);
        Status load_checkpoint(std::string checkpoint_folder);

        // TODO fix inputs & outputs to tensors or mats
        std::vector<float> predict(std::vector<int> input);
        Status fit(std::vector<std::vector<int>> train_date, std::vector<std::vector<float>> train_labels, float learning_rate, int epochs, std::string checkpoints_folder);
        
    private:
        std::string _name;
    
        std::vector<Layer> _layers;

        float _calculate_loss(std::vector<float> predicted_labels, std::vector<float> actual_labels);
        Status update_weights(float loss, float learning_rate);
    };
}
