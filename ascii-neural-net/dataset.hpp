#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>

#include <boost/algorithm/string.hpp>

#include <ascii-neural-net/types.hpp>

#define INPUT_LENGTH 48
#define OUTPUT_LENGTH 35

namespace ann
{
    class DatasetItem
    {
    public:
        DatasetItem(const std::vector<Scalar>& input, const std::vector<Scalar>& expected_output);

        const RowVector& input() const
        {
            return _input;
        }

        const RowVector& expected_output() const
        {
            return _expected_output;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    private:
        RowVector _input;
        RowVector _expected_output;
    };
    
    class Dataset
    {
    public:
        Dataset(std::string filename);

        void shuffle();
        DatasetItem* next();

        bool epoch_end()
        {
            return _items.size() == _next_item;
        }

        void reset_epoch()
        {
            _next_item = 0;
        }

    private:
        std::vector<DatasetItem> _items;
        int _next_item;
    };
}