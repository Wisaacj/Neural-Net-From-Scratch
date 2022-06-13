//
// Created by Will on 10/06/2022.
//

#ifndef MNIST_NEURAL_NET_ACTIVATIONFUNCTIONS_H
#define MNIST_NEURAL_NET_ACTIVATIONFUNCTIONS_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
 * Activation function: ReLU
 */
double ReLU(double x);

MatrixXd Softmax(MatrixXd Z);

/*
 * Derivative of ReLU activation function
 */
double ReLUDerivative(double x);

#endif //MNIST_NEURAL_NET_ACTIVATIONFUNCTIONS_H
