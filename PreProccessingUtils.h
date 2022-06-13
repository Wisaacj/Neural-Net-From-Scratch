//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_PREPROCCESSINGUTILS_H
#define MNIST_NEURAL_NET_PREPROCCESSINGUTILS_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

tuple<VectorXd, MatrixXd> splitLabelsFromFeatures(const MatrixXd& A);

MatrixXd oneHotEncode(const VectorXd& labels);

#endif //MNIST_NEURAL_NET_PREPROCCESSINGUTILS_H
