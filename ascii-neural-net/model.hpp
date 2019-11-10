#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ascii-neural-net/dataset.hpp>
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


        Status fit(Dataset& data, Scalar learning_rate, int epochs, std::string checkpoints_folder);
        
    private:
        std::string _name;
    
        std::vector<std::unique_ptr<Layer>> _layers;

        RowVector _output;

        Status _forward(const RowVector& input);
        Status _backprop(const RowVector& expected_output);
        Status _update(Scalar learning_rate);
    };
}
