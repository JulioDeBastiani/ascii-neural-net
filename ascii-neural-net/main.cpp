#include <iostream>
#include <string>

#include <ascii-neural-net/model.hpp>
#include <ascii-neural-net/dataset.hpp>
#include <ascii-neural-net/metrics.hpp>

int main()
{
    const std::string model_name = "model-a";

    std::cout << "Model name: \"" << model_name << "\"" << std::endl;

    ann::Model model(model_name);
    auto status = model.load_model("/home/julio/Documents/ucs/ascii-neural-net/models/" + model_name + ".ann");

    ann::Dataset dataset("/home/julio/Documents/ucs/ascii-neural-net/EntrancesAndExits.txt");
    dataset.shuffle();

    status = model.fit(dataset, 1e-1, 1000, "/home/julio/Documents/ucs/ascii-neural-net/checkpoints");

    if (status.err())
    {
        std::cout << "failed" << std::endl;
        return 1;
    }

    // model.load_checkpoint("/home/julio/Documents/ucs/ascii-neural-net/checkpoints");

    ann::Dataset test_dataset("/home/julio/Documents/ucs/ascii-neural-net/testDataset.txt");
    model.eval(test_dataset, true);
    test_dataset.reset_epoch();
    ann::create_confusion_matrix(model, test_dataset);

    return 0;
}