//
// Created by Will on 09/06/2022.
//

#ifndef MNIST_NEURAL_NET_MULTILAYEREDPERCEPTRONUTILS_H
#define MNIST_NEURAL_NET_MULTILAYEREDPERCEPTRONUTILS_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> initialiseWeightsAndBiases() {
    // MatrixXd::Random() generates a random matrix of elements in range(-1, 1)
    MatrixXd w_1 = MatrixXd::Random(10, 784);
    MatrixXd b_1 = MatrixXd::Random(10, 1);
    MatrixXd w_2 = MatrixXd::Random(10, 10);
    MatrixXd b_2 = MatrixXd::Random(10, 1);
    return make_tuple(w_1, b_1, w_2, b_2);
}

MatrixXd relu(const MatrixXd& Z) {

}

MatrixXd softmax(const MatrixXd& Z) {

}

#endif //MNIST_NEURAL_NET_MULTILAYEREDPERCEPTRONUTILS_H
