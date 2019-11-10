#include <ascii-neural-net/dataset.hpp>

namespace ann
{
    DatasetItem::DatasetItem(const std::vector<Scalar>& input, const std::vector<Scalar>& expected_output)
    {
        _input.resize(INPUT_LENGTH);
            
        for (int i = 0; i < INPUT_LENGTH; i++)
        {
            _input(i, 0) = input[i];
        }

        _expected_output.resize(OUTPUT_LENGTH);
        
        for (int i = 0; i < OUTPUT_LENGTH; i++)
        {
            _expected_output(i, 0) = expected_output[i];
        }
    }

    Dataset::Dataset(std::string filename)
    {
        std::ifstream file(filename);

        if (!file.is_open())
        {
            std::cout << "could not open file \"" << filename << "\"" << std::endl;
            return;
        }

        std::string line;

        std::vector<Scalar> tinvec;
        tinvec.reserve(INPUT_LENGTH);

        std::vector<Scalar> toutvec;
        toutvec.reserve(OUTPUT_LENGTH);

        std::vector<std::string> tokens;

        while (getline(file, line))
        {
            tinvec.clear();
            toutvec.clear();
            tokens.clear();
                
            boost::split(tokens, line, [](char c){ return c == ' '; });

            if (tokens.size() != 2)
            {
                std::cout << "error in line \"" << line << "\"" << std::endl;
                return;
            }

            for (char c: tokens[0])
            {
                tinvec.push_back(c - '0');
            }

            for (char c: tokens[1])
            {
                toutvec.push_back(c - '0');
            }
                
            _items.push_back(DatasetItem(tinvec, toutvec));
        }

        _next_item = 0;
    }

    void Dataset::shuffle()
    {
        auto rng = std::default_random_engine {};
        std::shuffle(_items.begin(), _items.end(), rng);
    }

    DatasetItem* Dataset::next()
    {
        if (_items.size() == _next_item)
            return nullptr;

        auto ptr = &_items[_next_item];
        _next_item++;
        return ptr;
    }
}