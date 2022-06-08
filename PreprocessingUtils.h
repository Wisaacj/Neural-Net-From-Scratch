//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_PREPROCESSINGUTILS_H
#define MNIST_NEURAL_NET_PREPROCESSINGUTILS_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

tuple<VectorXd, MatrixXd> splitLabelsFromFeatures(MatrixXd A) {
    VectorXd labels = A(all, 0);
    MatrixXd features = A(all, seqN(1, last));

    return make_tuple(labels, features);
}

#endif //MNIST_NEURAL_NET_PREPROCESSINGUTILS_H
