//
// Created by Will on 13/06/2022.
//

#include <Eigen/Dense>
#include "ActivationFunctions.h"
#include "EigenUtils.h"

/*
 * Activation function: ReLU
 */
double ReLU(double x) {
    // If x > 0 then return x, else 0
    return (x > 0 ? x : 0);
}

/*
 * Derivative of ReLU activation function
 */
double ReLUDerivative(double x) {
    return (x > 0 ? 1 : 0);
}

/*
 * Activation function: Softmax
 * [ Do not want to use pass-by-reference here ]
 */
MatrixXd Softmax(MatrixXd Z) {
    // Iterating column-wise
    for(int i = 0; i < Z.cols(); i++) {
        Z.col(i) = Z.col(i).unaryExpr(&eigenExp) / Z.col(i).unaryExpr(&eigenExp).sum();
    }
    return Z;
}