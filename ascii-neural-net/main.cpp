#include <iostream>
#include <string>

#include <ascii-neural-net/model.hpp>

int main()
{
    const std::string model_name = "model-a";

    std::cout << "Model name: \"" << model_name << "\"" << std::endl;

    ann::Model model(model_name);
    auto status = model.load_model("../models/" + model_name + ".ann");
    status = model.save_checkpoint("../checkpoints");
    status = model.load_checkpoint("../checkpoints");

    if (status.err())
        std::cout << "sigh" << std::endl;

    return 0;
}