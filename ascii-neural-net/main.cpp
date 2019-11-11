#include <iostream>
#include <string>

#include <ascii-neural-net/model.hpp>
#include <ascii-neural-net/dataset.hpp>

int main()
{
    const std::string model_name = "model-a";

    std::cout << "Model name: \"" << model_name << "\"" << std::endl;

    ann::Model model(model_name);
    auto status = model.load_model("/home/julio/Documents/ucs/ascii-neural-net/models/" + model_name + ".ann");

    ann::Dataset dataset("/home/julio/Documents/ucs/ascii-neural-net/EntrancesAndExits.txt");
    dataset.shuffle();

    status = model.fit(dataset, 3e-1, 1000, "/home/julio/Documents/ucs/ascii-neural-net/checkpoints");

    if (status.err())
    {
        std::cout << "failed" << std::endl;
        return 1;
    }

    ann::Dataset test_dataset("/home/julio/Documents/ucs/ascii-neural-net/testDataset.txt");

    return 0;
}