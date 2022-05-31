//
// Created by Will on 31/05/2022.
//

#include "CSVReader.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

/* Implementation file for CSVReader */

CSVReader::CSVReader() = default; // Default constructor

CSVReader::~CSVReader() = default; // Default de-constructor

/*
 * Credit:
 * https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
 *
 * Thoughts:
 * 1. Might need to change the datatype of values vector to int?
 */
template<typename M>
M CSVReader::load_csv(const std::string &path) {
    std::ifstream indata;
    indata.open(path);

    std::string line;
    std::vector<double> values;

    int rows = 0; // Might need to change this to an unsigned int
    while(std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            // Parsing cells interpreting their content as a floating-point numbers
            values.push_back(std::stod(cell));
        }
        ++rows;
    }

    return Map<const Matrix<typename M::Scalar
}