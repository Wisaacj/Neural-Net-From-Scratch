//
// Created by Will on 10/06/2022.
//

#include <Eigen/Dense>
#include "MLP.h"
#include "ActivationFunctions.h"

using namespace std;
using namespace Eigen;

MLP::MLP() {
    this->initialiseParameters();
}

MLP::~MLP() = default;

void MLP::initialiseParameters() {
    this->w_1 = MatrixXd::Random(10, 784);
    this->b_1 = MatrixXd::Random(10, 1);
    this->w_2 = MatrixXd::Random(10, 10);
    this->b_2 = MatrixXd::Random(10, 1);
}

void MLP::train(const MatrixXd &X, const MatrixXd &y, double learningRate, int epochs) {

}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> MLP::forwardPropagation(const MatrixXd &X) {
    // Calculate first hidden-layer (unactivated)
    MatrixXd Z_1 = (this->w_1 * X) + this->b_1;
    // Apply ReLU activation function (activated)
    MatrixXd A_1 = Z_1.unaryExpr(&ReLU);
    // Calculate output-layer (unactivated)
    MatrixXd Z_2 = (this->w_2 * A_1) + this->b_1;
    // Apply softmax activation function (activated)
    MatrixXd A_2 = Softmax(Z_2);

    return make_tuple(Z_1, A_1, Z_2, A_2);
}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>
MLP::backwardPropagation(const MatrixXd &X, const MatrixXd &y, const MatrixXd &A_2, const MatrixXd &A_1, const MatrixXd &Z_1) {
    MatrixXd dZ_2 = A_2 - y;
    MatrixXd dW_2 = (dZ_2 * A_1.transpose()) * 1/X.cols(); // X.cols() is the number of instances
    MatrixXd db_2 = dZ_2.rowwise().sum() * 1/X.cols(); // Average (row-wise) error

    MatrixXd dZ_1 = (this->w_2.transpose() * dZ_2).cwiseProduct(Z_1.unaryExpr(&ReLUDerivative)); // cwiseProduct() is coefficient-wise
    MatrixXd dW_1 = (dZ_1 * X.transpose()) * (1/X.cols());
    MatrixXd db_1 = dZ_1.rowwise().sum() * 1/X.cols();

    return make_tuple(dW_1, db_1, dW_2, db_2);
}

void MLP::updateParameters(const MatrixXd &dW_1, const MatrixXd &db_1, const MatrixXd &dW_2, const MatrixXd &db_2, double learningRate) {

}

MatrixXd MLP::predict(const MatrixXd &X) {
    return X;
}
