//
// Created by Will on 10/06/2022.
//

#ifndef MNIST_NEURAL_NET_ACTIVATIONFUNCTIONS_H
#define MNIST_NEURAL_NET_ACTIVATIONFUNCTIONS_H

#include <Eigen/Dense>
#include "EigenUtils.h"

using namespace std;
using namespace Eigen;

/*
 * Activation function: ReLU
 */
double ReLU(double x) {
    // If x > 0 then return x, else 0
    return (x > 0 ? x : 0);
}

/*
 * Activation function: Softmax
 * [ Do not want to use pass-by-reference here ]
 */
//MatrixXd Softmax(MatrixXd Z) {
//    // Iterating column-wise
//    const auto zColwise = Z.colwise();
//    for_each(zColwise.begin(), zColwise.end(), [](const auto &column){
//        return column.unaryExpr(&EigenExp) / column.unaryExpr(&EigenExp).sum();
//    });
//    return Z;
//}

MatrixXd Softmax(MatrixXd Z) {
    return Z;
}

/*
 * Derivative of ReLU activation function
 */
double ReLUDerivative(double x) {
    return (x > 0 ? 1 : 0);
}

#endif //MNIST_NEURAL_NET_ACTIVATIONFUNCTIONS_H
