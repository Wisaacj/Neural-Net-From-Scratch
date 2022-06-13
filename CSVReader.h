//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_CSVREADER_H
#define MNIST_NEURAL_NET_CSVREADER_H

#include <Eigen/Dense>
#include <fstream>
#include <vector>

using namespace Eigen;

template<typename M>
M load_csv (const std::string & path);

#endif //MNIST_NEURAL_NET_CSVREADER_H
