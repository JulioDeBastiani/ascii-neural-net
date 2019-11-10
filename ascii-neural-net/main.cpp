#include <iostream>
#include <string>

#include <ascii-neural-net/model.hpp>
#include <ascii-neural-net/dataset.hpp>

int main()
{
    const std::string model_name = "model-a";

    std::cout << "Model name: \"" << model_name << "\"" << std::endl;

    ann::Model model(model_name);
    auto status = model.load_model("../models/" + model_name + ".ann");

    ann::Dataset dataset("../EntrancesAndExits.txt");
    dataset.shuffle();

    return 0;
}