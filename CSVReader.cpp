//
// Created by Will on 13/06/2022.
//

#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include "CSVReader.h"

using namespace Eigen;

/*
 * Credit:
 * https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
 */
template<typename M>
M load_csv (const std::string & path) {
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