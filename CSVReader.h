//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_CSVREADER_H
#define MNIST_NEURAL_NET_CSVREADER_H

#include <Eigen/Dense>
#include <fstream>
#include <vector>

using namespace Eigen;

/*
 * Credit:
 * https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
 */
template<typename M>
M load_csv (const std::string &path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;

    std::vector<double> values;
    unsigned int rows = 0;

    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }

    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

#endif //MNIST_NEURAL_NET_CSVREADER_H
