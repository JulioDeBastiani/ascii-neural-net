#pragma once

#include <ascii-neural-net/dataset.hpp>
#include <ascii-neural-net/model.hpp>
#include <ascii-neural-net/types.hpp>

namespace ann
{
    Matrix create_confusion_matrix(Model& model, Dataset& dataset);
}