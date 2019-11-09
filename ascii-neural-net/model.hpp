#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ascii-neural-net/layers.hpp>
#include <ascii-neural-net/status.hpp>

namespace ann
{
    // TODO rule of five
    class Model
    {
    public:
        Model(std::string name);

        Status load_model(std::string filename);
        Status save_checkpoint(std::string checkpoint_folder);
        Status load_checkpoint(std::string checkpoint_folder);

        void forward(const RowVector& input);

        // std::vector<float> predict(std::vector<int> input);
        // Status fit(std::vector<std::vector<int>> train_date, std::vector<std::vector<float>> train_labels, float learning_rate, int epochs, std::string checkpoints_folder);
        
    private:
        std::string _name;
    
        std::vector<std::unique_ptr<Layer>> _layers;

        float _calculate_loss(std::vector<float> predicted_labels, std::vector<float> actual_labels);
        Status update_weights(float loss, float learning_rate);
    };
}
