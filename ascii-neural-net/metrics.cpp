#include <ascii-neural-net/metrics.hpp>

namespace ann
{
    int largmax(const RowVector& vec)
    {
        int rows = vec.rows();
        int max_index = 0;
        Scalar  max_val = vec(0, 0);

        for (int i = 1; i < rows; i++)
        {
            Scalar val = vec(i, 0);

            if (val > max_val)
            {
                max_val = val;
                max_index = i;
            }
        }

        return max_index;
    }

    Matrix create_confusion_matrix(Model& model, Dataset& dataset)
    {
        Matrix conf_mat = Matrix::Zero(OUTPUT_LENGTH, OUTPUT_LENGTH);

        while (!dataset.epoch_end())
        {
            auto item = dataset.next();

            auto output = model.predict(item->input());
            auto predicted = largmax(output);
            auto truth = largmax(item->expected_output());
            
            if (item->expected_output()(truth) == 0)
                continue;

            conf_mat(truth, predicted) += 1;
        }

        std::ofstream file;
        file.open("../confusion-matrix.txt", std::ios_base::out);
        file << conf_mat;
        file.close();

        return conf_mat;
    }
}